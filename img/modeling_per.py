import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import time

from tqdm import tqdm
from sklearn.model_selection import train_test_split

##부모파일에 있는 py파일 읽기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('load_data.py'))))
 
from load_data import load_data_using_multi_process
from model import model_lgbm
from make_roc_curve import make_roc_curve

from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score



obj = 'fan'
model0 = '0'
model_6 = '_6'
model6 = '6'
id_name = 'id_00'
sr = 16000
target_sr = 125

# modeling_Model함수 사용 
## aa = modeling_Model('fan', 'id_02', 16000, 125)

def modeling_Model(obj:str,color:str, id_name:str, sr:int, target_sr:int):
    '''
    model별로 [0, _6, 6]노이즈를 합쳐서 모델링하기
        input
            obj : Object-ex)'fan'
            color : ROC커브 색깔
            id_name : 합치고 싶은 id_name - ex)'id_00'
            sr : sample rate - ex)16000
            target_sr : down sampling rate - ex)125
        output
            make_roc_curve함수
                - acc_score
                - recall_score
                - precision_score
                - f1_score
                - roc auc value
    '''

    Noise0 = '0'
    Noise_6 = '_6'
    Noise6 = '6'
    
    print('#'*10, '데이터 로드', '#'*10)
    ##fan_0_
    fan_0_id_00_path = '/data/time_series/'+obj+'/'+Noise0+'/' + id_name
    fan_0_id_00_files = glob.glob(fan_0_id_00_path + '/*/*')
    print(obj+'_'+Noise0+'_id_00_files 개수 : ', len(fan_0_id_00_files))
    print('\n')

    ##fan__6_
    fan__6_id_00_path = '/data/time_series/'+obj+'/'+Noise_6+'/' + id_name
    fan__6_id_00_files = glob.glob(fan__6_id_00_path + '/*/*')
    print(obj+'_'+Noise_6+'_id_00_files 개수 : ', len(fan__6_id_00_files))
    print('\n')

    ##fan_0_
    fan_6_id_00_path = '/data/time_series/'+obj+'/'+Noise6+'/' + id_name
    fan_6_id_00_files = glob.glob(fan_6_id_00_path + '/*/*')
    print(obj+'_'+Noise6+'_id_00_files 개수 : ', len(fan_6_id_00_files))
    print('\n')

    print('#'*10, 'Multi Processing', '#'*10)

    fan_0_ = load_data_using_multi_process(files = fan_0_id_00_files, sr = sr, target_sr = target_sr)
    fan__6_ = load_data_using_multi_process(files = fan__6_id_00_files, sr = sr, target_sr = target_sr)
    fan_6_= load_data_using_multi_process(files = fan_6_id_00_files, sr = sr, target_sr = target_sr)

    if (len(fan_0_[0]) == (target_sr*10 + 1)) & (len(fan__6_[0])==(target_sr*10 + 1)) & (len(fan_6_[0])==(target_sr*10 + 1)):
        fan_0_ = pd.DataFrame(fan_0_)
        fan__6_ = pd.DataFrame(fan__6_)
        fan_6_ = pd.DataFrame(fan_6_)

    df = pd.concat([fan_0_,fan__6_,fan_6_])

    if df.columns[-1] == target_sr*10 :
        df.rename(columns={target_sr*10:'label'}, inplace=True)

    df.reset_index(drop=True, inplace=True)
    
    print('\n')
    print('#'*10, 'Data Split', '#'*10, '\n')

    X = df.iloc[:, :target_sr*10]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify = y,
                                                        test_size = 0.2,
                                                        random_state = 42)
    print('X_train 의 크기 : ', X_train.shape)
    print('X_test 의 크기 : ', X_test.shape)
    print('y_train 의 크기 : ', y_train.shape)
    print('y_test 의 크기 : ', y_test.shape)
    print('\n')
    
    print('#'*10, 'Modeling', '#'*10)

    model = model_lgbm(df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    print('#'*10, 'ROC커브 그리기', '#'*10)
    return make_roc_curve(y_test, y_pred, y_pred_prob, color, id_name)

#########################################################

obj = 'fan'
Noise = '0'
sr = 16000
target_sr = 125

# modeling_Noise함수 사용
## bb = modeling_Noise('fan', '0', 16000, 125)

def modeling_Noise(obj:str,color:str, noise:str, sr:int, target_sr:int):
    '''
    Noise별로 모델링을 진행함
    input :
        - obj : Object - ex)'fan'
        - color : ROC커브 색깔
        - model : 모델링하고 싶은 노이즈 - ex)'0'
        - sr : sampling rate - ex)16000
        - target_sr : down sampling rate - ex)125
    output :
        make_roc_curve함수
            - acc_score
            - recall_score
            - precision_score
            - f1_score
            - roc auc value
    '''
    
    print('#'*10, '데이터 경로 불러오기', '#'*10)
    fan_0_id_00_path = '/data/time_series/'+obj+'/'+ noise
    all_fan_0 = glob.glob(fan_0_id_00_path + '/*/*/*')
    print(obj+'_'+noise+' 개수 : ', len(all_fan_0))
    print('\n')

    print('#'*10, 'Multi-Processing', '#'*10)
    fan_all_model = load_data_using_multi_process(files = all_fan_0, sr = sr, target_sr = target_sr)

    if len(fan_all_model[0]) == (target_sr*10)+1:
        df = pd.DataFrame(fan_all_model)

    if df.columns[-1] == target_sr*10 :
        df.rename(columns={target_sr*10:'label'}, inplace=True)
    df.reset_index(drop=True, inplace=True)   
    print('\n')

    print('#'*10, 'Data Split', '#'*10)
    X = df.iloc[:, :target_sr*10]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify = y,
                                                        test_size = 0.2,
                                                        random_state = 42)
    print('X_train 의 크기 : ', X_train.shape)
    print('X_test 의 크기 : ', X_test.shape)
    print('y_train 의 크기 : ', y_train.shape)
    print('y_test 의 크기 : ', y_test.shape)
    print('\n')

    print('#'*10, 'Modeling', '#'*10)
    model = model_lgbm(df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    print('#'*10, 'ROC_CURVE', '#'*10)
    return make_roc_curve(y_test, y_pred, y_pred_prob, color, noise)
    
    
####################################################

obj = 'fan'
sr = 16000
target_sr = 125

# modeling_Object함수 사용
## cc = modeling_Object('fan', 16000, 125)

def modeling_Object(obj:str, sr:int, target_sr:int):
    '''
    원하는 Object를 전체로 모델링 돌리기
        input :
            - obj 모델링 돌리고 싶은 Object - ex)'fan'
            - sr : sampling rate - ex)16000
            - target_sr : down sampling rate - ex)125
        output : 
            make_roc_curve함수
                - acc_score
                - recall_score
                - precision_score
                - f1_score
                - roc auc value
    '''

    print('#'*10, '파일경로 불러오기', '#'*10)
    fan_path = '/data/time_series/' + obj
    fan_files = glob.glob(fan_path + '/*/*/*/*')
    print('\n')

    print('#'*10, 'Multi-Processing', '#'*10)
    result = load_data_using_multi_process(files = fan_files, sr = sr, target_sr = target_sr)

    if len(result[0]) == (target_sr*10)+1:
        df = pd.DataFrame(result)

    if df.columns[-1] == target_sr*10 :
        df.rename(columns={target_sr*10:'label'}, inplace=True)
    print('\n')

    print('#'*10, 'Data Split', '#'*10)
    X = df.iloc[:, :target_sr*10]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify = y,
                                                        test_size = 0.2,
                                                        random_state = 42)
    print('X_train 의 크기 : ', X_train.shape)
    print('X_test 의 크기 : ', X_test.shape)
    print('y_train 의 크기 : ', y_train.shape)
    print('y_test 의 크기 : ', y_test.shape)
    print('\n')

    print('#'*10, 'Modeling', '#'*10)
    model = model_lgbm(df)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    print('\n')
    
    print('#'*10, 'ROC_CURVE', '#'*10)
    return make_roc_curve(y_test, y_pred, y_pred_prob)



































