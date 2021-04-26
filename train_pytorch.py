import argparse
from importlib import import_module
from pathlib import Path

import warnings
warnings.filterwarnings(action='ignore')

import os
import sys
sys.path.append(os.path.abspath('./KoBERT-Transformers'))
import glob
import pickle
import re
import json
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
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

def set_neptune(save_dir, args):
    ns = neptune.init(project_qualified_name = 'cha-no/KLUE', api_token = API_TOKEN)
    
    # Create experiment
    neptune.create_experiment(name = f'{save_dir}')

    neptune.append_tag(args.dataset)
    neptune.append_tag(args.criterion)
    neptune.append_tag(f'max_len-{args.max_len}')
    neptune.append_tag(args.model)
    neptune.append_tag(f'num_hidden_layers-{args.num_hidden_layers}')
    neptune.append_tag(args.optimizer)
    neptune.append_tag(f'preprocess-{args.preprocess}')
    neptune.append_tag(args.tokenize)
    neptune.append_tag(args.scheduler)
    neptune.append_tag(f'seed-{args.seed}')
    neptune.log_metric('batch_size', args.batch_size)
    neptune.log_metric('num_epochs', args.epochs)


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

def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    
    s_dir = args.model + str(args.num_hidden_layers) + '-' + args.preprocess + '-epoch' + str(args.epochs) + \
            '-' + args.criterion + '-' + args.scheduler + '-' + args.optimizer + '-' + args.dataset + '-' + args.tokenize
    
    if args.name:
        s_dir += '-' + args.name
    save_dir = increment_path(os.path.join(model_dir, s_dir))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("This notebook use [%s]."%(device))

    # load model and tokenizer
    MODEL_NAME = args.model
    if MODEL_NAME == "monologg/kobert":
        tokenizer = KoBertTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    dataset = load_data("/opt/ml/input/data/train/train.tsv")
    labels = dataset['label'].values

    # setting model hyperparameter
    bert_config = BertConfig.from_pretrained(MODEL_NAME)
    bert_config.num_labels = args.num_labels
    bert_config.num_hidden_layers = args.num_hidden_layers
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config = bert_config)
    model.dropout = nn.Dropout(p = args.drop)
    model.to(device)

    summary(model)

    # loss & optimizer
    if args.criterion == 'f1' or args.criterion == 'label_smoothing' or args.criterion == 'f1cross':
        criterion = create_criterion(args.criterion, classes = args.num_labels, smoothing = 0.1)
    else:
        criterion = create_criterion(args.criterion)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == 'AdamP':
        optimizer = AdamP(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            optimizer_grouped_parameters,
            lr = args.lr,
        )
    
    # logging
    logger = SummaryWriter(log_dir = save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    set_neptune(save_dir, args)

    # preprocess dataset
    if args.preprocess != 'no':
        pre_module = getattr(import_module("preprocess"), args.preprocess)
        dataset = pre_module(dataset, model, tokenizer)

    # train, val split
    kfold = StratifiedKFold(n_splits = 5)

    for train_idx, val_idx in kfold.split(dataset, labels):
        train_dataset, val_dataset = dataset.loc[train_idx], dataset.loc[val_idx]
        break
    
    tok_module = getattr(import_module("load_data"), args.tokenize)

    train_tokenized = tok_module(train_dataset, tokenizer, max_len = args.max_len)
    val_tokenized = tok_module(val_dataset, tokenizer, max_len = args.max_len)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(train_tokenized, train_dataset['label'].reset_index(drop = 'index'))
    RE_val_dataset = RE_Dataset(val_tokenized, val_dataset['label'].reset_index(drop = 'index'))

    train_loader = DataLoader(
        RE_train_dataset,
        batch_size = args.batch_size,
        num_workers = 4,
        shuffle = True,
        pin_memory = use_cuda,
    )

    val_loader = DataLoader(
        RE_val_dataset,
        batch_size = 12,
        num_workers = 1,
        shuffle = False,
        pin_memory = use_cuda,
    )

    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)
    elif args.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 5)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    elif args.scheduler == 'cosine_warmup':
        t_total = len(train_loader) * args.epochs
        warmup_step = int(t_total * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)
    else:
        scheduler = None

    print("Training Start!!!")

    best_val_acc = 0
    best_val_loss = np.inf
    
    for epoch in range(args.epochs):
        # train loop
        model.train()

        train_loss, train_acc = AverageMeter(), AverageMeter()

        for idx, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            try:
                inputs, token_types, attention_mask, labels = train_batch.values()
                inputs = inputs.to(device)
                token_types = token_types.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outs = model(input_ids = inputs, token_type_ids = token_types, attention_mask = attention_mask)
            except:
                inputs, attention_mask, labels = train_batch.values()
                inputs = inputs.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outs = model(input_ids = inputs, attention_mask = attention_mask)

            preds = torch.argmax(outs.logits, dim =-1)
            loss = criterion(outs.logits, labels)
            acc = (preds == labels).sum().item() / len(labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

            if scheduler:
                scheduler.step()

            neptune.log_metric('learning_rate', get_lr(optimizer))

            train_loss.update(loss.item(), len(labels))
            train_acc.update(acc, len(labels))

            if (idx + 1) % args.log_interval == 0:
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss.avg:.4f} || training accuracy {train_acc.avg:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss.avg, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc.avg, epoch * len(train_loader) + idx)

        neptune.log_metric(f'Train_loss', train_loss.avg)
        neptune.log_metric(f'Train_avg', train_acc.avg)
        neptune.log_metric('learning_rate', current_lr)

        val_loss, val_acc = AverageMeter(), AverageMeter()
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            
            for val_batch in val_loader:
                try:
                    inputs, token_types, attention_mask, labels = val_batch.values()
                    inputs = inputs.to(device)
                    token_types = token_types.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    outs = model(input_ids = inputs, token_type_ids = token_types, attention_mask = attention_mask)
                except:
                    inputs, attention_mask, labels = val_batch.values()
                    inputs = inputs.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    outs = model(input_ids = inputs, attention_mask = attention_mask)

                preds = torch.argmax(outs.logits, dim = -1)
                loss = criterion(outs.logits, labels)
                acc = (preds == labels).sum().item() / len(labels)

                val_loss.update(loss.item(), len(labels))
                val_acc.update(acc, len(labels))
            
            if val_acc.avg > best_val_acc:
                print(f"New best model for val acc : {val_acc.avg:4.2%}! saving the best model..")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc.avg
                best_val_loss = min(best_val_loss, val_loss.avg)
            
            print(
                f"[Val] acc : {val_acc.avg:4.2%}, loss : {val_loss.avg:.4f} || "
                f"best acc : {best_val_acc:4.2%}, best loss : {best_val_loss:.4f}"
            )
            logger.add_scalar("Val/loss", val_loss.avg, epoch)
            logger.add_scalar("Val/accuracy", val_acc.avg, epoch)
            neptune.log_metric(f'Val_loss', val_loss.avg)
            neptune.log_metric(f'Val_avg', val_acc.avg)
            
            print()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--seed', type = int, default = 2021, help = 'random seed (defualt : 2021)')
    parser.add_argument('--dataset', type = str, default = 'original', help = 'dataset (defualt : original)')
    parser.add_argument('--model', type = str, default = 'bert-base-multilingual-cased', help = 'model type (default : bert-base-multilingual-cased)')    
    parser.add_argument("--num_labels", type = int, default = 42, help = 'num_labels')
    parser.add_argument("--num_hidden_layers", type = int, default = 3, help = 'num_hidden_layers')    
    parser.add_argument("--drop", type = float, default = 0.5, help = 'dropout')    
    parser.add_argument("--warmup_ratio", type = float, default = 0.01, help = 'warmup_ratio')    
    parser.add_argument("--max_len", type = int, default = 200, help = 'max_len')
    parser.add_argument("--preprocess", type = str, default = 'no', help = 'preprocess')    
    parser.add_argument("--tokenize", type = str, default = 'tokenized_dataset', help = 'tokenized_dataset')    
    parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs to train (default : 1)')
    parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'optimizer type (default : Adam)')
    parser.add_argument('--lr', type = float, default = 5e-5, help = 'learning rate (default : 3e-4)')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default : 5e-4)')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'input batch size for training (default : 16)')
    parser.add_argument('--valid_batch_size', type = int, default = 12, help = 'input batch size for validing (default : 12)')
    parser.add_argument('--patience', type = int, default = 5, help = 'patience (default : 5)')
    parser.add_argument('--val_ratio', type = float, default = 0.2, help = 'ratio for validaton (default : 0.2)')
    parser.add_argument('--criterion', type = str, default = 'cross_entropy', help = 'criterion type (default : CrossEntropyLoss)')
    parser.add_argument('--scheduler', type = str, default = 'no', help = 'learning rate scheduler (default: no)')
    parser.add_argument('--log_interval', type = int, default = 50, help = 'how many batches to wait before logging training status')
    parser.add_argument('--name', default = None, help = 'model save at {SM_MODEL_DIR}/{name}')
    
    # Container environment
    parser.add_argument('--data_dir', type = str, default = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model_dir', type = str, default = os.environ.get('SM_MODEL_DIR', './models'))

    args = parser.parse_args()
    print(args)

    train(args.data_dir, args.model_dir, args)