# KLUE

# 목차

- [프로젝트 소개](#프로젝트-소개)
- [Problem](#problem)
  - [회고록](#회고록)

## 프로젝트 소개

### 문장 내 개체간 관계 추출

- 관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다.
- 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다.

<img src = "https://user-images.githubusercontent.com/59329586/122316470-e7eae480-cf56-11eb-9835-0ad65a978041.png" width="70%" height="35%">

- 위 그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.


### 목적

- 목적은 문장, 엔티티, 관계에 대한 정보를 통해 문장과 엔티티 사이의 관계를 추론하는 모델을 학습시킵니다.

- input
```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
entity 1: 썬 마이크로시스템즈
entity 2: 오라클

```
- output : 단체:별칭



### 평가방법

- 모델은 **Accuracy** 로 평가됩니다.

### 데이터셋

- 데이터셋의 예시입니다.

<img src = "https://user-images.githubusercontent.com/59329586/122322094-dce88200-cf5f-11eb-9f79-3a5ec2113892.png" width="70%" height="35%">

## Problem

- Input의 변경

기존의 Input
```
ex) 이순신 장군은 조선 출신이다.
entity1 : 이순신, entity2 : 조선

[CLS] 이순신 [SEP] 조선 [SEP] 이순신 장군은 조선 출신이다. [SEP]
```

엔티티 강조를 위해 엔티티 앞뒤로 토큰을 붙입니다.

```
[CLS] [ENT1] 이순신 [/ENT1] 장군은 [ENT2] 조선 [/ENT2] 출신이다. [SEP]
```

BERT형식으로 Input을 변경했습니다.

```
[CLS] [ENT1] 이순신 [/ENT1] 장군은 [ENT2] 조선 [/ENT2] 출신이다. [SEP] 이 문장에서 [ENT1] 이순신 [/ENT1] 와 [ENT2] 조선 [/ENT2] 의 관계는 무엇인가? [SEP]
```

### [회고록](https://www.notion.so/0f1fd5626ec54c748d0b8bb399f913bd)

## usage
```python
## train
python train.py --epochs 10 --max_len 300 --model xlm-roberta-large --batch_size 16 --preprocess convert9 --tokenize tokenized_dataset1 --drop 0.5 --seed 2022

## inference
python inference.py --model xlm-roberta-large12-convert9-epoch10-linear-tokenized_dataset1-300-2022/checkpoint-3100 --token xlm-roberta-large --max_len 300 --name xlm-roberta-large12-convert9-epoch10-linear-tokenized_dataset1-300-2023 --preprocess convert9 --tokenize tokenized_dataset1 --seed 2022
```
