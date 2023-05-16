#make EDA class
# !pip install librosa
# !python3 -m pip install --upgrade pip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os 
import glob


import librosa
import librosa.core
import librosa.feature
import logging

from tqdm import tqdm
from sklearn import metrics

# 사용자 운영체제 확인
import platform
platform.system()

# 운영체제별 한글 폰트 설정
if platform.system() == 'Darwin': # Mac 환경 폰트 설정
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows 환경 폰트 설정
    plt.rc('font', family='Malgun Gothic')

plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정


# 글씨 선명하게 출력하는 설정
# %config InlineBackend.figure_format = 'retina'



class EDA:
    def __init__(self, path:str):
        #path : '/data/time_series/fan/'
        self.path = path
        self.pile = path.split('/')[-1]
        print('good')
        
    
    def make_path_dic(self):
        '''
        make_path_dic : 각 아이디별, 정상비정상별, 각 wav파일의 경로를 담은 딕셔너리
        (추가기능) - 각 파일별 개수 시각화

            path : 객체까지 있어야함
            ex) '/project/timeseries_project/data/fan'
        '''
        data_list = glob.glob(self.path + '/*')

        path_dic = {}

        for ind, idpath in enumerate(data_list): #각 ID별로 딕셔너리 만들기 
            id_name = idpath[-4:]
            # {'id_00': {}, 'id_02': {}, 'id_06': {}, 'id_04': {}}
            path_dic[id_name] = {}

            noabn = glob.glob(idpath+'/*') #정상비정상신호 각각 path
            for noa in noabn :
                noa_name = noa.split('/')[-1] #normal/abnormal 이름 저장
                path_dic[id_name][noa_name] = glob.glob(noa + '/*')

        ## 각 파일별 개수 시각화
        fig = plt.figure(figsize=(15,5))
        axs = fig.subplots(1,4)

        for ii, (id_int, no_abn) in enumerate(path_dic.items()) :
            print('\n')
            print(id_int)

            y_ax = []
            x_ax = []
            for k, v in no_abn.items() :
                print(k, ':' , len(v))
                x_ax.append(k)
                y_ax.append(len(v))
            axs[ii].set_title('{}'.format(id_int))
            axs[ii].bar(x_ax, y_ax)          
        path_dic = path_dic
        return path_dic
    
    #fan_dic = make_path_dic(path)
    
    def make_visualization(self,dic, file_name:str, nor_or_abnor:str, sr:int):
        '''#file_path:str
        하나의 id, 정상 혹은 비정상 신호를 합쳐서 그래프 그리기
            file_path : normal 혹은 abnormal까지 나와야함
            ex) fan_dic['d_00']['normal']
            =>>
                ['/project/timeseries_project/data/fan/id_00/normal/00000899.wav',
                 '/project/timeseries_project/data/fan/id_00/normal/00000419.wav',
                 '/project/timeseries_project/data/fan/id_00/normal/00000254.wav',
                 '/project/timeseries_project/data/fan/id_00/normal/00000388.wav',
                 ... ]
            sr : 1초당 몇개로 끊을건지 (ex : sr=100 : 1초당 100개씩 바꿈)
        '''    

        fig = plt.figure(figsize=(70,15))

        y_mean_li = []
        y_std_li = []

        for audio_path in dic[file_name][nor_or_abnor]:
            y, sr = librosa.load(audio_path, sr=sr) 
            #y = sorted(y) #이거삭제하면 원본임
            y_mean_li.append(y.mean())
            y_std_li.append(y.std())

            _=plt.plot(range(sr*10), y, alpha = 0.2, color='grey')
        title = '{}'.format(file_name) + ' ' + '{}'.format(nor_or_abnor)
        _=plt.title('{}'.format(title), fontsize=40)
        _=plt.xlabel('sr')
        _=plt.xticks(fontsize=30)
        _=plt.yticks(fontsize=30)

        _=plt.plot([0,1000], [np.mean(y_mean_li),np.mean(y_mean_li)], color='g', linewidth=10)
        _=plt.show()
        print('평균 : ', np.mean(y_mean_li))
        print('편차 : ', np.mean(y_std_li)) 
        print('good')
        
    def make_sorted_plot(self, dic, file_name:str, nor_or_abnor:str, sr:int):
        fig = plt.figure(figsize=(70,15))
        if nor_or_abnor == 'normal':
            ##normal : 빨간색
            normal_path = dic[file_name][nor_or_abnor]

            for audio_path in normal_path:
                y, sr = librosa.load(audio_path, sr=sr) 
                y = sorted(y)



                _=plt.plot(range(sr*10), y, alpha = 0.2, color='red')
            _=plt.xlabel('sr')
            _=plt.xticks(fontsize=30)
            _=plt.yticks(fontsize=30)  

        elif nor_or_abnor == 'abnormal':
            ##abnormal : 파란색
            abnormal_path = dic[file_name][nor_or_abnor]

            for audio_path in abnormal_path:
                y, sr = librosa.load(audio_path, sr=sr) 
                y = sorted(y)



                _=plt.plot(range(sr*10), y, alpha = 0.2, color='blue')
            _=plt.xlabel('sr')
            _=plt.xticks(fontsize=30)
            _=plt.yticks(fontsize=30) 

        else :
            ##both
            normal_path = dic[file_name]['normal']

            for audio_path in normal_path:
                y, sr = librosa.load(audio_path, sr=sr) 
                y = sorted(y)
                _=plt.plot(range(sr*10), y, alpha = 0.2, color='red')

            abnormal_path = fan_dic[file_name]['abnormal']

            for audio_path in abnormal_path:
                y, sr = librosa.load(audio_path, sr=sr) 
                y = sorted(y)



                _=plt.plot(range(sr*10), y, alpha = 0.2, color='blue')


            _=plt.xlabel('sr')
            _=plt.xticks(fontsize=30)
            _=plt.yticks(fontsize=30) 
        print('good')
        
    
    def make_KDE(self, dic, file_name:str, nor_or_abnor:str, sr:int):
        '''
        모든 파일 각각 KDE를 겹쳐서 그리는함수
            file_name : 파일이름 (ex)'d_00')
            nor_or_abnor : 'normal' or 'abnormal' or 'both'
        '''
        if nor_or_abnor != 'both': #'normal' or 'abnormal'
            audio_path = dic[file_name][nor_or_abnor]
            fig = plt.figure(figsize=(70,30))

            for ii, path in enumerate(audio_path) :
                y, sr = librosa.load(path, sr=sr)  #sr : 초당샘플개수
                if nor_or_abnor == 'normal' :
                    if ii == 0 : # labeling
                        sns.kdeplot(y, shade=False, alpha=0, color='red', label= nor_or_abnor)
                        _=plt.legend(['RED : {}'.format(nor_or_abnor)], fontsize=100)
                    sns.kdeplot(y, shade=False, alpha=0.2, color='red')            
                else : # 'abnormal'
                    if ii == 0 : # labeling
                        sns.kdeplot(y, shade=False, alpha=0, color='blue', label= nor_or_abnor)
                        _=plt.legend(['BLUE : {}'.format(nor_or_abnor)], fontsize=100)
                    sns.kdeplot(y, shade=False, alpha=0.2, color='blue')    
            _=plt.xticks(fontsize=40)
            _=plt.yticks(fontsize=40)
            _=plt.ylabel('Density', fontsize=40)
            _=plt.title('KDE Plot : "{} ({})"'.format(file_name, nor_or_abnor), fontsize=40)
        else : # 'both'
            audio_path_normal = dic[file_name]['normal']
            audio_path_abnormal = dic[file_name]['abnormal']

            fig = plt.figure(figsize=(70,30))

            for ii, path_normal in enumerate(audio_path_normal):
                y_normal,sr = librosa.load(path_normal, sr=sr)
                if ii == 0 :# labeling
                    sns.kdeplot(y_normal, shade=False, alpha=0.1, color='red', label= 'normal')
                sns.kdeplot(y_normal, shade=False, alpha=0.2, color='red')

            for ii, path_abnormal in enumerate(audio_path_abnormal):
                y_abnormal,sr = librosa.load(path_abnormal, sr=sr)
                if ii == 0 :# labeling
                    sns.kdeplot(y_abnormal, shade=False, alpha=0, color='blue', label= 'abnormal')
                sns.kdeplot(y_abnormal, shade=False, alpha=0.2, color='blue')
            _=plt.legend(['RED : normal', 'BLUE : abnormal'], fontsize=100)
            _=plt.xticks(fontsize=40)
            _=plt.yticks(fontsize=40)
            _=plt.ylabel('Density', fontsize=40)
            _=plt.title('KDE Plot : "{} (both)"'.format(file_name), fontsize=40)
        _=plt.show()
        print('good')
