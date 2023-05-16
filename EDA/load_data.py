import librosa
import numpy as np
from tqdm import tqdm
import time

stride = 100
sr = 16000

# 데이터 로드
def sampling_abs_max(file, target_sr, sr):
    if 'abnormal' in file: # 비정상 데이터
        cls = 1
    else:                # 정상데이터
        cls = 0

    data_lst, _ = librosa.load(file, sr = sr)
    if len(data_lst) != sr*10:
        print("data 길이가 다릅니다. -> ", len(data_lst))
    
    stride = int(sr/target_sr) # 16,000/160 = 100

# 데이터 전처리
    result = []
    stride_data = []
    for i, row in enumerate(data_lst):

        if i % stride != 0:
            stride_data.append(abs(row))
        else:
            stride_data.append(abs(row))
            result.append(np.max(stride_data))
            stride_data=[]

    result.append(cls) # 마지막 컬럼에 클래스 부여
    return result

def sampling_lib(file, sr, target_sr, sampling_method): # target_sr은 초당 몇개를 샘플할건지 개수 정해야함 -> 160
    import lazy_loader as lazy
    resampy = lazy.load("resampy")
    
    result = []
    if 'abnormal' in file: # 비정상 데이터
        cls = 1
    else:                # 정상데이터
        cls = 0

    data_lst, _ = librosa.load(file, sr = sr)
    if len(data_lst) != sr*10:
        print("data 길이가 다릅니다. -> ", len(data_lst))

    data_lst = librosa.resample(data_lst, orig_sr = sr, target_sr = target_sr, res_type = sampling_method)
    data_lst = np.append(data_lst, cls)
    result.append(data_lst)
    return result

# sampling_lib함수는 값하나를 받아 처리하는 함수


def load_data_using_multi_process(files, sr, target_sr):
    sampling_lst = ['abs_max(our_custom)',
                    'kaiser_best',
                    'kaiser_fast',
                    'fft',
                    'scipy',
                    'polyphase',
                    'linear',
                    'zero_order_hold',
                    'sinc_best',
                    'sinc_medium' ,
                    'sinc_fastest',
                    'soxr_vhq',
                    'soxr_hq',
                    'soxr_mq',
                    'soxr_lq',
                    'soxr_hq',
                    'soxr_qq']
    print('\n아래 리스트에서 샘플링 기법을 선택해주세요. \n', sampling_lst)
    sampling_method = input('선택한 기법: ')
    
    #### 멀티프로세싱에 필요한 속성 생성 ####
    if sampling_method not in sampling_lst:
        print('sampling_method 오타 났잖아 정신 차리자')
        return 0
    
    elif sampling_method == 'abs_max(our_custom)':
        load_data_def = sampling_abs_max

    else:
        load_data_def = sampling_lib
        
    args_list=[]
    for file in files:   
        if sampling_method == 'abs_max(our_custom)':
            args_list.append([file, target_sr, sr])
        else:
            args_list.append([file, sr, target_sr, sampling_method])
    
    #### 멀티 프로세싱 ####
    from multiprocessing import Pool
    import time
    import lazy_loader as lazy
    resampy = lazy.load("resampy")
    start_time = time.time()
    result=[]
    processes=30
    print(f'\n멀티 프로세싱을 시작합니다. 현재 코어 {processes}개 사용 중. \n1.다른 사람과 동시에 돌리지 마세요.\n2.함부로 진행중에 중단 하지마세요.')
    p = Pool(processes) # 몇개의 코어를 이용할 것인지 설정
    for data in p.starmap(load_data_def, args_list): # 각 코어에 입력값들을 병렬 처리
        if sampling_method == 'abs_max(our_custom)':
            result.append(data)
        else:
            result+=data
    p.close() # 멀티 프로세싱 종료
    p.join()

    end_time = time.time()
    print('--- 걸린시간: {} ---'.format(end_time - start_time))
    return result



























    