import pandas as pd
import numpy as np
import string
from khaiii import KhaiiiApi
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import pickle

from catboost import CatBoostClassifier

# 분류예측 함수 정의
def QnA_Classification(input_txt):
    data = pd.DataFrame({"text": [input_txt]})
    data["txt_len"] = data["text"].apply(len)

    # 토큰화 및 불용어 처리
    data[["nouns", "morphs", "pos"]] = data['text'].apply(lambda x: pd.Series(khaiii_analysis(x)))
    data["nouns"] = data["nouns"].apply(lambda x: [word for word in x if word not in stopwords])

    # 질문별 알파벳, 특수문자 빈도수 확인 및 데이터 프레임 저장
    data["alpha_cnt"] = data["text"].apply(lambda x: len([char for char in x if char.lower() in string.ascii_lowercase]))
    data["punc_cnt"] = data["text"].apply(lambda x: len([char for char in x if char in string.punctuation]))

    # 벡터화
    CountVectorizer()
    data_nouns = [' '.join(text) for text in data["nouns"]]
    
    with open('./count_vectorizer.pickle', 'rb') as handle:
        vec_nouns = pickle.load(handle)
    
    input_vec_nouns = vec_nouns.transform(data_nouns)

    # 나머지 숫자형 변수들을 희소 행렬로 변환
    data_sparse = pd.DataFrame(data, columns=['txt_len', 'alpha_cnt', 'punc_cnt']).values

    # 최종 데이터 구성
    input_data = hstack([input_vec_nouns, data_sparse])

    # 모델 불러오기
    model = CatBoostClassifier().load_model('catboost_model.cbm')
    prediction = model.predict(input_data)
    reversed_label_dict = {
        0: '코드',
        1: '웹',
        2: '이론',
        3: '시스템 운영',
        4: '원격'
    }

    prediction = model.predict(input_data)
    return reversed_label_dict[int(prediction[0])]

# 토큰화 함수 정의
def khaiii_analysis(sentence):
    nouns, morphs, pos = [], [], []
    for word in KhaiiiApi().analyze(sentence):
        word_nouns = []
        word_morphs = []
        for morph in word.morphs:
            if morph.tag.startswith('N') and morph.tag != "NNBC":
                word_nouns.append(morph.lex)
            word_morphs.append((morph.lex, morph.tag))
        morphs.append(word_morphs)
        pos.extend(word_morphs)
        nouns.extend(word_nouns)
    return (nouns, morphs, pos)

# 불용어 사전을 이용하여 불용어 불러오기
with open("./Stopwords_ko.txt", "r", encoding="utf-8") as f:
    stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]

# 데이터 입력
input_txt = input()
predicted_label = QnA_Classification(input_txt)
print(f'질문 유형: {predicted_label}')