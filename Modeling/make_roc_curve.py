import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, recall_score, precision_score


# def make_roc_curve(y_test, y_pred, y_pred_prob, color):
def make_roc_curve(y_test, y_pred, y_pred_prob, color, label):
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred) 
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    
#     plt.figure(figsize = (7, 7))
#     plt.plot(fpr, tpr, color = color, label = 'ROC curve')
    plt.plot(fpr, tpr, color = color, label = label)
#     plt.plot([0, 1], [0, 1], color = 'blue', label = 'y = x') # y = x 직선 표시

#     plt.xlabel('FP Rate')
#     plt.ylabel('TP Rate')

#     plt.legend() # 그래프 라벨 표시

#     plt.show()
    print(f'acc_score : {round(acc, 3)}')
    print(f'recall_score : {round(recall, 3)}')
    print(f'precision_score : {round(precision, 3)}')
    print(f'f1_score : {round(f1, 3)}')
    
    print(f'roc auc value : {round(roc_auc_score(y_test, y_pred_prob[:,1]), 3)}')
    return round(acc, 3), round(recall, 3), round(precision, 3), round(f1, 3), round(roc_auc_score(y_test, y_pred_prob[:,1]), 3)