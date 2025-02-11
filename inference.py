import argparse
from importlib import import_module
from pathlib import Path

import os
import pickle
import random

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from torch.utils.data import DataLoader
import torch

from load_data import *
from preprocess import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def inference(model, tokenized_sent, device, batch_size):
    logits = []
    predictions = []

    dataloader = DataLoader(tokenized_sent, batch_size = batch_size, shuffle = False)
    model.eval()

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            if 'token_type_ids' in data.keys():
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device),
                    token_type_ids=data['token_type_ids'].to(device)
                )
            else:
                outputs = model(
                    input_ids=data['input_ids'].to(device),
                    attention_mask=data['attention_mask'].to(device)
                )

        _logits = outputs[0].detach().cpu().numpy()      
        _predictions = np.argmax(_logits, axis=-1)

        logits.append(_logits)
        predictions.extend(_predictions.ravel())

    return np.concatenate(logits), np.array(predictions)

def load_test_dataset(dataset_dir, model, tokenizer, args):
    test_dataset = load_data(dataset_dir)
    # preprocess dataset
    if args.preprocess != 'no':
        pre_module = getattr(import_module("preprocess"), args.preprocess)
        test_dataset = pre_module(test_dataset, model, tokenizer)

    test_label = test_dataset['label'].values
    # tokenizing dataset
    tok_module = getattr(import_module("load_data"), args.tokenize)

    tokenized_test = tok_module(test_dataset, tokenizer, args.max_len)
    return tokenized_test, test_label

def main(args):
    """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    seed_everything(args.seed)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load tokenizer
    TOK_NAME = args.token
    if TOK_NAME.startswith('xlm'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(TOK_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

    # load my model
    MODEL_NAME = os.path.join(args.model_dir, args.model) # model dir.
    if TOK_NAME.startswith('xlm'):
        model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME)
    else:
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    

    # load test datset
    test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
    test_dataset, test_label = load_test_dataset(test_dataset_dir, model, tokenizer, args)
    test_dataset = RE_Dataset(test_dataset ,test_label)

    model.to(device)

    # predict answer
    batch_size = args.batch_size
    logits, pred_answer = inference(model, test_dataset, device, batch_size)
    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

    output = pd.DataFrame(pred_answer, columns=['pred'])
    save_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    output.to_csv(os.path.join(save_dir, f'{args.name}.csv'), index=False)
    np.save(os.path.join(save_dir, r'logits.npy'), logits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type = int, default = 2021, help = 'random seed (defualt : 2021)')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'input batch size for validing (default : 32)')
    parser.add_argument('--model', type = str, default = "test/checkpoint-500")
    parser.add_argument("--num_labels", type = int, default = 42, help = 'num_labels')    
    parser.add_argument("--num_hidden_layers", type = int, default = 12, help = 'num_hidden_layers')    
    parser.add_argument("--tokenize", type = str, default = 'tokenized_dataset', help = 'tokenized_dataset')    
    parser.add_argument("--preprocess", type = str, default = 'no', help = 'preprocess')    
    parser.add_argument('--token', type = str, default = "bert-base-multilingual-cased" , help = 'token (default : bert-base-multilingual-cased)')
    parser.add_argument("--max_len", type = int, default = 200, help = 'max_len')    
    parser.add_argument('--name', type = str, default = "output") 

    # Container environment
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './models'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    print(args)
    main(args)
  
