from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def model_lgbm(data): # input datatype is DataFrame
    normal = len( data[data.iloc[:, -1] == 0] )
    abnormal = len( data[data.iloc[:, -1] == 1] )
    print(f'normal : {normal}\nabnormal : {abnormal}')
    
    scale_pos_weight = round( normal / abnormal ,2)
    print(f'scale_pos_weight is {scale_pos_weight}.')
    
    params = {  'random_state' : 42,
                'scale_pos_weight' : scale_pos_weight,
                'learning_rate' : 0.1, 
                'num_iterations' : 1000,
                'max_depth' : 4,
                'n_jobs' : 30,
                'boost_from_average' : False,
                'objective' : 'binary' }
    
    model = LGBMClassifier( **params )
    print(f'Model is ready to running.')
    
    return model