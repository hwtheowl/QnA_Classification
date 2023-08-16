# QnA_Classification
AI 교육에서 교육생이 하게되는 1대1 문의 내용을 바탕으로 해당 문의글의 유형을 자동으로 분류하는 모델 개발 프로젝트
<br>
<br>
<br>

## 목차
- [프로젝트 개요](#프로젝트-개요)
- [프로젝트 목적](#프로젝트-목적)
- [개발 환경](#개발-환경)
- [사용 기술](#사용-기술)
- [데이터 소개](#데이터-소개)
- [데이터 분석 및 전처리](#데이터-분석-및-전처리)
- [모델링 및 평가](#모델링-및-평가)
- [결론](#결론)
- [프로젝트를 통해 느낀점](#프로젝트를-통해-느낀점)
<br>
<br>

## 프로젝트 개요
**유형 분류**

사용자의 질문을 바탕으로 해당 질문의 유형을 파악
<br>
<br>
<br>

## 프로젝트 목적
**왜?**

AI교육을 통해 배운것을 활용하여 서비스를 만드는 것이 핵심이었습니다.

문의 게시판에 수많은 질문이 들어왔을때, 질문자에게 질문유형을 고르게 하는 것이 아닌 자동으로 선택되는 방법을 통해 사용자 편의를 증진 시키고 질문 유형에 따라 담당자 배정을 빠르게 진행 할 수 있게 하는 목적이 있습니다.
<br>
<br>
<br>

## 개발 환경
- Visual Studio Code
- Github
- Colab
- Ubuntu 22.04
<br>
<br>

## 사용 기술
- Python
- TensorFlow
- Khaiii
- CountVectorizer
- Catboost
<br>
<br>

## 데이터 소개
**데이터 구조 [row 3706, column 2] / [label 6가지]**
![데이터 예시 표](https://github.com/hwtheowl/QnA_Classification/assets/132368135/fcb61589-14a7-4e0d-b836-1a8140582bc8)
<br>
<br>
<br>
<br>

## 데이터 분석 및 전처리
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/aaec668c-91a5-4220-a2ed-0b1a889d4147)

데이터 특성상 코드 관련 질문이 약 43%에 해당
<br>
<br>
<br>

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/75e6e85d-fef5-482a-81ab-18b131d79f0a)

형태소 분석기는 카카오에서 개발한 Khaiii large를 사용
<br>
<br>
<br>

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/272567f6-172c-4787-bc2b-8baf70a40911)

코드 질문의 경우 알파벳이 많을 것으로 판단하여 알파벳 갯수를 표기한 컬럼 추가
<br>
<br>
<br>

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/64da72d2-dda5-4449-a8f2-76ed5bc9b0ed)

예상대로 알파벳 갯수에 따른 질문 유형이 두 분류(코드+웹 / 이외)로 나뉘는 경향을 확인
<br>
<br>
<br>

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/665a1c1e-b895-4fbb-91b4-ca12e08eb3f6)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/0b936261-a0e7-46ea-93fb-5dd16ea5ded5)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/36fd8292-537a-40b0-a275-2698c8312da9)

불용어 처리 데이터와 불용어 처리를 하지 않은 데이터로 나누어 비교해본 시각화 자료

불용어 처리를 하지 않은 데이터의 경우 "것"이라는 단어가 압도적으로 많은 것으로 파악

향후 모델 성능에 영향이 있을지 확인하기 위해 2가지 버전으로 데이터셋 분리
<br>
<br>
<br>

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/284d687a-2a83-4562-821a-ce9173a6414a)

학습의 목적이 있기때문에 백터화의 경우 여러 방향으로 시도
<br>
<br>
<br>

## 모델링 및 평가
**df1: 불용어 처리를 하지 않은 데이터셋**

**df2: 불용어 처리를 완료한 데이터셋**
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/1312dbc0-3795-4861-bdcb-63bdcd962320)

ML의 경우, RandomForest, LightGBM, XGBoost, CatBoost, SVM으로 진행해보았고 SVM의 경우 모든 부분의 성능이 문제 발생

나머지 결과값에서 성능이 좋은 XGBoost의 df1, df2의 CountVect 사용 데이터셋과 CatBoost도 XGBoost와 동일한 데이터셋을 사용하여 그리드 서치 시행

그리드 서치 시행후 가장 좋았던 CatBoost를 이용한 df2_cnt 데이터셋으로 학습한 모델을 ML 대표로 선정
<br>
<br>
<br>

**DL 1st Model**

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/9d9fbb36-e8fe-4d31-874e-2ca3972d93f1)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/a37af829-6d84-482e-88af-a6a041437701)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/d957fc11-edb8-448f-b704-fcb4374960ec)

<br>
<br>

**DL 2nd Model**

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/aac5e355-4fc0-4cb0-afa2-e3ec40f36fd5)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/b7b5b01d-f6a4-4859-b900-5c594bef6825)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/5e5eeb73-430a-47a4-9e8c-341406cededa)

DL의 경우 Dense layer, BatchNomalization, Dropout을 이용한 간단한 설계를 한 모델과 RNN을 이용한 모델 총 두가지 버전으로 만듬
DL 첫번째 모델 정확도 0.74, 두번째 모델 정확도 0.4717의 정확도를 가짐
<br>
<br>
<br>

## 결론
최종 정확도에서 가장 준수한 성능을 보인 **CatBoost**모델을 df2_cnt(불용어 처리를 완료, CountVect를 사용한 데이터셋)으로 학습한 모델 선정

![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/e3011e2e-9c46-43cb-a3a2-c5e6430c0e36)
![image](https://github.com/hwtheowl/QnA_Classification/assets/132368135/46a8289f-2ffd-404e-ae6c-e67c5f5853b7)

테스트 결과 입력값이 주어지면 해당 텍스트의 질문 유형이 나오도록 설계완료

<br>
<br>

## 프로젝트를 통해 느낀점
모델 설계에 대해 다시 생각해보는 프로젝트였습니다.

DL열풍에 나도 모르게 ML은 DL보다는 성능이 떨어진다라고 생각하였지만, 설계 및 학습에 따라서는 ML이 더 우수한 성능을 보이기도 한다는것을 제대로 느꼈습니다. 설계, 전처리, 학습 등 저의 실력이 부족하여 만족스러운 결과를 얻지는 못하였지만 그렇기에 얻은게 있다고 생각하고 추후에는 이번 프로젝트를 경험삼아 더 나은 결과물을 뽑아낼 수 있다고 생각합니다.

사실 토큰화 작업을 할 때, 환경이슈로 Khaiii를 사용하기 위해 굉장히 애를 먹었습니다. 그 외에도 계속되는 문제를 맞딱드리며 순탄하게 진행하지는 못했지만, 이런 문제들을 스스로 해결해나가며 문제 해결 능력을 기를 수 있었고 포기하지 않고 계속 해나감으로써 결과물을 만들수 있었습니다. 그리고 결과물이 나왔을때 "해냈다."라는 생각과 **힘들지만 코딩이 재밌다.** 라는 다시한번 되뇌었습니다.

부족한게 많지만 더 공부해가며 발전 할 수 있는 가능성을 보여준 프로젝트였습니다.
