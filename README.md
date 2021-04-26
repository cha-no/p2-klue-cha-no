# Pstage_02_KLUE_T1117_신찬호
* 자연어처리를 이용한 관계 추출

## training
### transformer 모듈 버전
* python train.py
### pytorch 버전
* python train_pytorch.py
* ex) python train.py --seed 2021 --epochs 10 --model xlm-roberta-large

### inference
* python inference.py --model [model_path]
* ex) python inference.py --model xlm-roberta-large/checkpoint-2000

### loss
* crossentropy, f1loss, focalloss, label_smoothing_loss가 있음

### load_data
* 데이터를 불러들여오고, tokenize하는 함수

### preprocess
* 데이터 전처리 함수

## usage

```python
## 학습 단일모델 최고 성능 0.792
python train.py --epochs 10 --max_len 300 --model xlm-roberta-large --batch_size 16 --preprocess convert9 --tokenize tokenized_dataset1 --drop 0.5 --seed 2022

## inference
python inference.py --model xlm-roberta-large12-convert9-epoch10-linear-tokenized_dataset1-300-2022/checkpoint-3100 --token xlm-roberta-large --max_len 300 --name xlm-roberta-large12-convert9-epoch10-linear-tokenized_dataset1-300-2023 --preprocess convert9 --tokenize tokenized_dataset1 --seed 2022
```