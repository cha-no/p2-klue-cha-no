import re
import random
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Mecab, Okt
from pororo import Pororo

with open('/opt/ml/불용어.txt', 'r', encoding = 'utf8') as f:
    STOPWORDS = f.read()
STOPWORDS = STOPWORDS.split('\n')

STOPWORDS1 = []
with open('/opt/ml/한국어불용어100.txt', 'r', encoding = 'utf8') as f:
    temp = f.read()

for stopword in temp.split('\n'):
    STOPWORDS1.append(stopword.split('\t')[0])

STOPWORDS = set(STOPWORDS + STOPWORDS1)

def insert_ent_tag(text, ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx):
    new_text = ''
    
    for i, ch in enumerate(text):
        if i == ent1_start_idx:
            new_text += ' [ENT1] '
            new_text += ch
            
            if ent1_start_idx == ent1_end_idx:
                new_text += ' [/ENT1] '
                
        elif i == ent1_end_idx:
            new_text += ch
            new_text += ' [/ENT1] '
            
        elif i == ent2_start_idx:
            new_text += ' [ENT2] '
            new_text += ch
            
            if ent2_start_idx == ent2_end_idx:
                new_text += ' [/ENT2] '
                
        elif i == ent2_end_idx:
            new_text += ch
            new_text += ' [/ENT2] '
            
        else:
            new_text += ch
    
    # entity index 위치 변경
    if ent1_start_idx < ent2_start_idx:
        ent1_end_idx += 2
        ent2_start_idx += 2
        ent2_end_idx += 4
    else:
        ent2_end_idx += 2
        ent1_start_idx += 2
        ent1_end_idx += 4

    return new_text, ent1_start_idx, ent1_end_idx, ent2_start_idx, ent2_end_idx

def add_sentence(sentence):
    add_entity01 = '[ENT1] ' + sentence['entity_01'] + ' [/ENT1]'
    add_entity02 = '[ENT2] ' + sentence['entity_02'] + ' [/ENT2]'
    add_sen = '이 문장에서 ' + add_entity01 + ' 와 ' + add_entity02 + '의 관계는 무엇인가?' + tokenizer.sep_token
    return add_sen

def convert1(dataset, model, tokenizer):
    r_pattern = "[\'\":;.,()~`/\\\]"
    mecab = Mecab()
    def _convert_sentence1(sentence):
        sentence = re.sub(r_pattern, ' ', sentence)
        sentence = mecab.morphs(sentence)
        sentence = [word for word in sentence if word not in STOPWORDS]
        return ' '.join(sentence)
    dataset['sentence'] = dataset['sentence'].apply(lambda x : _convert_sentence1(x))
    return dataset

def convert2(dataset, model, tokenizer):
    added_token_num = 0
    for i in tqdm(range(len(dataset))):
        added_token_num += tokenizer.add_tokens(list(dataset.loc[i, ['entity_01', 'entity_02']].values))
    
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    
    return dataset

def convert3(dataset, model, tokenizer):
    r_pattern = "[\'\":;.,()~`/\\\]"
    okt = Okt()
    def _convert_sentence3(sentence):
        sentence = re.sub(r_pattern, ' ', sentence)
        sentence = okt.morphs(sentence)
        sentence = [word for word in sentence if word not in STOPWORDS]
        return ' '.join(sentence)
    dataset['sentence'] = dataset['sentence'].apply(lambda x : _convert_sentence3(x))
    return dataset


def convert5(dataset, model, tokenizer):
    r_pattern = "[\'\":;.,~`/\\\]"
    okt = Okt()
    def _convert_sentence5(sentence):
        sentence = re.sub(r_pattern, ' ', sentence)
        sentence = okt.morphs(sentence)
        sentence = [word for word in sentence if word not in STOPWORDS]
        return ' '.join(sentence)
    dataset['sentence'] = dataset['sentence'].apply(lambda x : _convert_sentence5(x))
    return dataset

def convert6(dataset, model, tokenizer):
    okt = Okt()
    def _convert_sentence6(sentence):
        sentence = okt.morphs(sentence)
        return ' '.join(sentence)
    dataset['sentence'] = dataset['sentence'].apply(lambda x : _convert_sentence6(x))
    return dataset

def convert7(dataset, model, tokenizer):
    mecab = Mecab()
    def _convert_sentence7(sentence):
        sentence = mecab.morphs(sentence)
        return ' '.join(sentence)
    dataset['sentence'] = dataset['sentence'].apply(lambda x : _convert_sentence7(x))
    return dataset

def convert8(dataset, model, tokenizer):
    def augmentation(index):
        data = small_dataset.loc[index, 'sentence'].split()
        entity1, entity2 = small_dataset.loc[index, 'entity_01'], small_dataset.loc[index, 'entity_02']
        label = small_dataset.loc[index, 'label']
        count = 0
        total = 0
        while True:
            if count >= 5 or total >= 1500:
                break
            random.shuffle(data)
            sen = ' '.join(data)
            if entity1 in sen and entity2 in sen:
                small_dict['sentence'].append(sen)
                small_dict['entity_01'].append(entity1)
                small_dict['entity_02'].append(entity2)
                small_dict['label'].append(label)
                count += 1
            total += 1
    
    small_labels = dataset.groupby('label').count()[dataset.groupby('label').count()['sentence'] < 100].index
    small_dataset = dataset[dataset['label'].apply(lambda x : x in small_labels)].reset_index().drop(columns = 'index')
    small_dict = {'sentence' : [], 'entity_01' : [], 'entity_02' : [], 'label' : []}

    for ind in tqdm(small_dataset.index):
        augmentation(ind)
    
    small_aug = pd.DataFrame(small_dict)
    total_dataset = pd.concat([dataset, small_aug]).reset_index().drop(columns = 'index')
    return total_dataset

def convert9(dataset, model, tokenizer):
    dataset[['sentence', 'entity_01_start', 'entity_01_end', 'entity_02_start', 'entity_02_end']] = dataset.apply(lambda x: insert_ent_tag(x[1], int(x[3]), int(x[4]), int(x[6]), int(x[7])), axis=1, result_type='expand')
    
    added_token_num = tokenizer.add_tokens(['[ENT1]', '[/ENT1]', '[ENT2]', '[/ENT2]'])
    
    model.resize_token_embeddings(len(tokenizer))
    
    return dataset

def convert10(dataset, model, tokenizer):
    special_entity = ['#', '@', "+", '^', 'ORGANIZATION', 'ARTIFACT', 'QUANTITY', 'PERSON', 'CIVILIZATION', 'LOCATION', 'OCCUPATION', 'CITY', 'DATE', 'COUNTRY', 'STUDY_FIELD', 'O', 'TERM', 'EVENT', 'ANIMAL', 'TIME', 'THEORY', 'MATERIAL', 'PLANT', 'DISEASE']
    ner = Pororo(task = 'ner', lang = 'ko')
    
    def add_entity1(data):
        sentence, entity_01, entity_02, entity_01_start, entity_01_end, entity_02_start, entity_02_end = data['sentence'], data['entity_01'], data['entity_02'], data['entity_01_start'], data['entity_01_end'], data['entity_02_start'], data['entity_02_end']
        ner_01 = ' # ' + ner(entity_01)[0][1] + ' # '
        ner_02 = ' @ ' + ner(entity_02)[0][1] + ' @ '

        entity_01_start, entity_01_end = int(entity_01_start), int(entity_01_end)
        entity_02_start, entity_02_end = int(entity_02_start), int(entity_02_end)
        if entity_01_start < entity_02_start:
            sentence = sentence[:entity_01_start] + ' + ' + ner_01 + sentence[entity_01_start:entity_01_end + 1] + ' + ' + sentence[entity_01_end + 1:entity_02_start] + ' ^ ' + ner_02 + sentence[entity_02_start:entity_02_end + 1] + ' ^ ' + sentence[entity_02_end + 1:]
        else:
            sentence = sentence[:entity_02_start] + ' ^ ' + ner_02 + sentence[entity_02_start:entity_02_end + 1] + ' ^ ' + sentence[entity_02_end + 1:entity_01_start] + ' + ' + ner_01 + sentence[entity_01_start:entity_01_end + 1] + ' + ' + sentence[entity_01_end + 1:]
        return sentence
    
    dataset['sentence'] = dataset.apply(add_entity1, axis = 1)

    added_token_num = tokenizer.add_tokens(special_entity)
    
    model.resize_token_embeddings(len(tokenizer))
    
    return dataset

def convert11(dataset, model, tokenizer):
    special_entity = [
        '[/ENT2-DATE]',
         '[ENT2-STUDY_FIELD]',
         '[/ENT1-TIME]',
         '[/ENT2-CITY]',
         '[ENT1-ANIMAL]',
         '[ENT1-CIVILIZATION]',
         '[ENT2-DATE]',
         '[ENT1-LOCATION]',
         '[/ENT2-STUDY_FIELD]',
         '[ENT2-QUANTITY]',
         '[/ENT2-ORGANIZATION]',
         '[ENT1-EVENT]',
         '[ENT1-CITY]',
         '[/ENT1-EVENT]',
         '[/ENT2-LOCATION]',
         '[/ENT2-EVENT]',
         '[ENT1-ORGANIZATION]',
         '[/ENT1-ARTIFACT]',
         '[/ENT1-CIVILIZATION]',
         '[ENT1-OCCUPATION]',
         '[ENT1-PLANT]',
         '[ENT2-EVENT]',
         '[/ENT2-PERSON]',
         '[ENT2-ARTIFACT]',
         '[ENT1-DATE]',
         '[/ENT1-PERSON]',
         '[/ENT1-LOCATION]',
         '[ENT2-ORGANIZATION]',
         '[ENT2-THEORY]',
         '[/ENT1-OCCUPATION]',
         '[/ENT2-OCCUPATION]',
         '[/ENT1-STUDY_FIELD]',
         '[ENT2-MATERIAL]',
         '[ENT1-STUDY_FIELD]',
         '[ENT1-PERSON]',
         '[ENT1-QUANTITY]',
         '[ENT2-PERSON]',
         '[/ENT1-DATE]',
         '[/ENT1-PLANT]',
         '[/ENT2-COUNTRY]',
         '[ENT2-ANIMAL]',
         '[/ENT1-ANIMAL]',
         '[/ENT2-TIME]',
         '[ENT2-TIME]',
         '[/ENT2-ANIMAL]',
         '[ENT2-PLANT]',
         '[ENT1-MATERIAL]',
         '[ENT2-CIVILIZATION]',
         '[ENT2-O]',
         '[/ENT1-QUANTITY]',
         '[/ENT2-MATERIAL]',
         '[ENT1-O]',
         '[ENT2-OCCUPATION]',
         '[ENT2-DISEASE]',
         '[/ENT1-MATERIAL]',
         '[/ENT2-ARTIFACT]',
         '[ENT1-THEORY]',
         '[ENT1-COUNTRY]',
         '[/ENT1-COUNTRY]',
         '[/ENT2-TERM]',
         '[ENT1-ARTIFACT]',
         '[/ENT2-O]',
         '[/ENT2-DISEASE]',
         '[/ENT2-CIVILIZATION]',
         '[/ENT1-TERM]',
         '[/ENT2-PLANT]',
         '[/ENT1-THEORY]',
         '[ENT2-LOCATION]',
         '[/ENT2-THEORY]',
         '[ENT1-TIME]',
         '[ENT2-COUNTRY]',
         '[ENT1-TERM]',
         '[ENT2-CITY]',
         '[/ENT1-O]',
         '[ENT2-TERM]',
         '[/ENT1-ORGANIZATION]',
         '[/ENT2-QUANTITY]',
         '[/ENT1-CITY]'
    ]
    ner = Pororo(task = 'ner', lang = 'ko')
    
    def add_entity2(data):
        sentence, entity_01, entity_02, entity_01_start, entity_01_end, entity_02_start, entity_02_end = data['sentence'], data['entity_01'], data['entity_02'], data['entity_01_start'], data['entity_01_end'], data['entity_02_start'], data['entity_02_end']
        ner_01_start = '[ENT1-' + ner(entity_01)[0][1] + ']'
        ner_01_end = '[/ENT1-' + ner(entity_01)[0][1] + ']'
        ner_02_start = '[ENT2-' + ner(entity_02)[0][1] + ']'
        ner_02_end = '[/ENT2-' + ner(entity_02)[0][1] + ']'

        entity_01_start, entity_01_end = int(entity_01_start), int(entity_01_end)
        entity_02_start, entity_02_end = int(entity_02_start), int(entity_02_end)

        if entity_01_start < entity_02_start:
            sentence = sentence[:entity_01_start] + ner_01_start + sentence[entity_01_start:entity_01_end + 1] + ner_01_end + sentence[entity_01_end + 1:entity_02_start] + ner_02_start + sentence[entity_02_start:entity_02_end + 1] + ner_02_end + sentence[entity_02_end + 1:]
        else:
            sentence = sentence[:entity_02_start] + ner_02_start + sentence[entity_02_start:entity_02_end + 1] + ner_02_end + sentence[entity_02_end + 1:entity_01_start] + ner_01_start + sentence[entity_01_start:entity_01_end + 1] + ner_01_end + sentence[entity_01_end + 1:]

        return sentence    
    
    dataset['sentence'] = dataset.apply(add_entity2, axis = 1)

    added_token_num = tokenizer.add_tokens(special_entity)
    
    model.resize_token_embeddings(len(tokenizer))
    
    return dataset