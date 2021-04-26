import os
import pickle

from tqdm import tqdm
import pandas as pd

import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, labels):
        self.tokenized_dataset = tokenized_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
    label = []
    for i in dataset['relation']:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])
    
    dataset['label'] = label
    # out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
    return dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
    # load label_type, classes
    with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
        label_type = pickle.load(f)
    # load dataset
    dataset = pd.read_csv(dataset_dir, delimiter = '\t', header = None)
    dataset.columns = ['information', 'sentence', 'entity_01', 'entity_01_start', 'entity_01_end', 'entity_02', 'entity_02_start', 'entity_02_end', 'relation']
    # preprecessing dataset
    dataset = preprocessing_dataset(dataset, label_type)

    return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer, max_len):
    concat_entity = []
    n = len(dataset['entity_01'])
    for i in tqdm(dataset.index):
        e01, e02 = dataset.loc[i, 'entity_01'], dataset.loc[i, 'entity_02']
        temp = ''
        temp = e01 + tokenizer.sep_token + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = max_len,
        add_special_tokens = True,
    )
    return tokenized_sentences

def add_sentence(sentence):
    add_entity01 = '[ENT1] ' + sentence['entity_01'] + ' [/ENT1]'
    add_entity02 = '[ENT2] ' + sentence['entity_02'] + ' [/ENT2]'
    add_sen = '이 문장에서 ' + add_entity01 + ' 와 ' + add_entity02 + '의 관계는 무엇인가?'
    return add_sen

def tokenized_dataset1(dataset, tokenizer, max_len):
    dataset['add_sentence'] = dataset.apply(add_sentence, axis = 1)
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        list(dataset['add_sentence']),
        return_tensors = 'pt',
        padding = True,
        truncation = True,
        max_length = max_len,
        add_special_tokens = True,
    )
    return tokenized_sentences