import argparse
from importlib import import_module
from pathlib import Path

import warnings
warnings.filterwarnings(action='ignore')

import os
import glob
import sys
sys.path.append(os.path.abspath('./KoBERT-Transformers'))
import pickle
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from tokenization_kobert import KoBertTokenizer

from adamp import AdamP

#import neptune.new as neptune
import neptune

from loss import create_criterion, AverageMeter
from load_data import *
from preprocess import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
API_TOKEN = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMzBjNjlkYi02ZWQ0LTQ4NzktYWZjNC1jNjc4NTdmZDhiNzQifQ=='

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def set_neptune(save_dir, args):
    ns = neptune.init(project_qualified_name = 'cha-no/KLUE', api_token = API_TOKEN)
    
    # Create experiment
    neptune.create_experiment(name = f'{save_dir}')

    neptune.append_tag(f'max_len-{args.max_len}')
    neptune.append_tag(args.model)
    neptune.append_tag(f'num_hidden_layers-{args.num_hidden_layers}')
    neptune.append_tag(f'preprocess-{args.preprocess}')
    neptune.append_tag(args.tokenize)
    neptune.append_tag(args.scheduler)
    neptune.append_tag(f'seed-{args.seed}')
    neptune.append_tag(f'smoothing_factor-{args.smoothing_factor}')
    neptune.append_tag(f'warmup_steps-{args.warmup_steps}')
    neptune.log_metric('batch_size', args.batch_size)
    neptune.log_metric('num_epochs', args.epochs)

# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model_dir, args):
    seed_everything(args.seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("This notebook use [%s]."%(device))

    s_dir = args.model + str(args.num_hidden_layers) + '-' + args.preprocess + '-epoch' + str(args.epochs) + '-' + args.scheduler + '-' + args.tokenize + '-' + str(args.max_len) + '-' + str(args.seed)
    
    save_dir = increment_path(os.path.join(model_dir, s_dir))
    log_dir = increment_path(os.path.join('logs', s_dir))

    # load model and tokenizer
    MODEL_NAME = args.model
    if MODEL_NAME.startswith('xlm'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # set neptune
    set_neptune(save_dir, args)

    # load dataset
    dataset = load_data("/opt/ml/input/data/train/train.tsv")
    labels = dataset['label'].values

    # setting model hyperparameter
    if MODEL_NAME.startswith('xlm'):
        bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
    else:
        bert_config = BertConfig.from_pretrained(MODEL_NAME)
    
    bert_config.num_labels = args.num_labels
    bert_config.num_hidden_layers = args.num_hidden_layers

    if MODEL_NAME.startswith('xlm'):
        model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config = bert_config)
    else:
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config = bert_config) 
    
    if args.drop >= 0:
        model.dropout = nn.Dropout(p = args.drop)

    # preprocess dataset
    if args.preprocess != 'no':
        pre_module = getattr(import_module("preprocess"), args.preprocess)
        dataset = pre_module(dataset, model, tokenizer)

    # make dataset for pytorch.
    # train, val split
    
    train_dataset, val_dataset = train_test_split(dataset, test_size = args.val_ratio, random_state = args.seed)

    tok_module = getattr(import_module("load_data"), args.tokenize)

    train_tokenized = tok_module(train_dataset, tokenizer, max_len = args.max_len)
    val_tokenized = tok_module(val_dataset, tokenizer, max_len = args.max_len)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(train_tokenized, train_dataset['label'].reset_index(drop = 'index'))
    RE_val_dataset = RE_Dataset(val_tokenized, val_dataset['label'].reset_index(drop = 'index'))

    model.to(device)
    

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        seed = args.seed,
        output_dir = save_dir,                          # output directory
        save_total_limit = 2,                           # number of total save model.
        save_steps = args.save_steps,                   # model saving step.
        num_train_epochs = args.epochs,                 # total number of training epochs
        learning_rate = args.lr,                        # learning_rate
        per_device_train_batch_size = args.batch_size,  # batch size per device during training
        per_device_eval_batch_size = 16,                # batch size for evaluation
        lr_scheduler_type = args.scheduler,
        warmup_steps = args.warmup_steps,               # number of warmup steps for learning rate scheduler
        weight_decay = args.weight_decay,               # strength of weight decay
        logging_dir = log_dir,                          # directory for storing logs
        logging_steps = 100,                            # log saving step.
        evaluation_strategy = 'steps',                  # evaluation strategy to adopt during training
                                                        # `no`: No evaluation during training.
                                                        # `steps`: Evaluate every `eval_steps`.
                                                        # `epoch`: Evaluate every end of epoch.
        eval_steps = 100,                               # evaluation step.
        dataloader_num_workers = 4,
        label_smoothing_factor = args.smoothing_factor,
        load_best_model_at_end = True,
        metric_for_best_model = 'accuracy'
    )


    trainer = Trainer(
        model = model,                          # the instantiated 🤗 Transformers model to be trained
        args = training_args,                   # training arguments, defined above
        train_dataset = RE_train_dataset,       # training dataset
        eval_dataset = RE_val_dataset,          # evaluation dataset
        compute_metrics = compute_metrics       # define metrics function
    )

    # train model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--seed', type = int, default = 2021, help = 'random seed (defualt : 2021)')
    parser.add_argument('--model', type = str, default = 'bert-base-multilingual-cased', help = 'model type (default : bert-base-multilingual-cased)')    
    parser.add_argument("--num_labels", type = int, default = 42, help = 'num_labels')    
    parser.add_argument("--num_hidden_layers", type = int, default = 12, help = 'num_hidden_layers')    
    parser.add_argument("--drop", type = float, default = -1, help = 'dropout')    
    parser.add_argument("--max_len", type = int, default = 200, help = 'max_len')    
    parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs to train (default : 1)')
    parser.add_argument('--save_steps', type = int, default = 100, help = 'number of save_steps to train (default : 100)')
    parser.add_argument("--preprocess", type = str, default = 'no', help = 'preprocess')    
    parser.add_argument("--tokenize", type = str, default = 'tokenized_dataset', help = 'tokenized_dataset')    
    parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'optimizer type (default : Adam)')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'learning rate (default : 3e-4)')
    parser.add_argument('--smoothing_factor', type = float, default = 0.5, help = 'label smoothing factor (default : 0.5)')
    parser.add_argument('--weight_decay', type = float, default = 0.01, help = 'weight decay (default : 5e-4)')
    parser.add_argument('--warmup_steps', type = int, default = 300, help = 'warup steps (default : 300)')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'input batch size for training (default : 16)')
    parser.add_argument('--val_ratio', type = float, default = 0.2, help = 'ratio for validaton (default : 0.2)')
    parser.add_argument('--scheduler', type = str, default = 'linear', help = 'learning rate scheduler (default: linear)')
    parser.add_argument('--name', default = 'exp', help = 'model save at {SM_MODEL_DIR}/{name}')
    
    # Container environment
    parser.add_argument('--model_dir', type = str, default = os.environ.get('SM_MODEL_DIR', './models'))

    args = parser.parse_args()
    print(args)

    train(args.model_dir, args)