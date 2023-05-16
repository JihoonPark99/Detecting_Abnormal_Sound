import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


## input : 
resample = 'kaiser_best'
# data




def whole_best_vis(data, resample:str):
    '''
    모든 모델링 결과들을 시각화하는 함수
        input : - data : 데이터프레임
                - resample : resampling type
        output : 가장 수치가 높았던 값들 데이터프레임 2개
    
    '''

    ID_li = ['id_00', 'id_02', 'id_04', 'id_06']
    color_li = ['r', 'y', 'g', 'b']
    factor = ['acc', 'recall', 'precision', 'auc_score', 'f1']


    fig = plt.figure(figsize=(30,20), dpi=400)
    axs = fig.subplots(3,2)

    # date = data[data['sampling_way'] == 'kaiser_best']
    val_dic = {}
    ind_dic = {}
    for ind, fac in enumerate(factor):
        val_dic[fac] = []
        ind_dic[fac] = []
        for ID, color in zip(ID_li, color_li):
            con_id_ = data['Model'] == ID

            X1 = range(len(data[con_id_]['target_sr'].values))
            Y = data[con_id_]


            maxval = np.argmax(Y[fac])
            val_dic[fac].append(Y[fac].iloc[maxval])
            ind_dic[fac].append(data[con_id_]['target_sr'].iloc[maxval])

            row = ind // 2 
            col = ind % 2 

            sns.pointplot(x="target_sr", y=fac,data=Y, color=color, label='{}'.format(ID), ax=axs[row, col])
#             _=axs[row, col].plot(X1,Y, color=color ,label='{}'.format(ID))
            _=axs[row, col].axvline(maxval, color=color, linestyle='--', linewidth=2)
            _=axs[row, col].set_title('{}'.format(fac), fontsize=20)
            _=axs[row, col].set_xticks(X1)
            _=axs[row, col].set_xticklabels(data[con_id_]['target_sr'], fontsize=20, rotation=45)
            



        # 범례 지정하기
        # 박스형태 범례
        variable_x = mpatches.Patch(color='r',label='id_00')
        variable_y = mpatches.Patch(color='y',label='id_02')
        variable_z = mpatches.Patch(color='g',label='id_04')
        variable_w = mpatches.Patch(color='b',label='id_06')


        #범례 나타내기
        _=axs[row, col].legend(handles=[variable_x, variable_y, variable_z,variable_w], loc='best', fontsize=10)
#         _=axs[row, col].legend(loc='best', fontsize=10)
        _=axs[row, col].set_ylim(0,1.2)


    X = np.arange(len(val_dic['acc']))

    for ind, fac in enumerate(factor):
        axs[2,1].bar(X+(ind*0.1), val_dic[fac], width=0.15, align='center', label='{}'.format(fac))

        for ii, index in enumerate(ind_dic[fac]):
            x_pos = X[ii] + (ind * 0.1)
            y_pos = val_dic[fac][ii]
            axs[2,1].text(x_pos, y_pos, '{}'.format(index), ha='left', va='bottom', fontsize=15)

    _= axs[2,1].set_xticks(X+0.2)
    _= axs[2,1].set_xticklabels(['id_00', 'id_02', 'id_04', 'id_06'], fontsize=20)
    _= axs[2,1].legend(loc='best', fontsize=20)
    _= axs[2,1].set_title('Every Max Values', fontsize=20)


    _=fig.suptitle('{}'.format(resample), fontsize=30)

    fig.tight_layout()

    val_df = pd.DataFrame(val_dic, index = ID_li)
    ind_df = pd.DataFrame(ind_dic, index = ID_li)

    return val_df, ind_df