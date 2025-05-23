
import os
from math import sqrt
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix,roc_curve
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,average_precision_score,accuracy_score,matthews_corrcoef
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score,mean_absolute_error,mean_absolute_percentage_error

parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
def onehot(labels,n_class):
    """print(onehot(np.array([0,1,2]),3))"""
    onehot = np.zeros((labels.shape[-1], n_class))
    for i, value in enumerate(labels):
        onehot[i, value] = 1
    return onehot

def de_onehot(labels):
    return np.argmax(labels, axis=1)


def onehot_preds(preds):
    arr_complement = 1 - preds
    preds = np.column_stack((arr_complement, preds))
    return preds


def Micro_OvR_AUC(labels, preds):
    fpr, tpr, _ = roc_curve(labels.ravel(), preds.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def Macro_OvR_AUC(labels, preds):
    n_classes = labels.shape[1]
    fpr={}
    tpr={}
    roc_auc={}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    return fpr_grid, mean_tpr, auc(fpr_grid, mean_tpr)

def Weighted_OvR_AUC(labels, preds, weighted_method="amount_divided"):
    n_classes = labels.shape[1]
    ls_num_equal1 = np.array([np.sum(labels[:,i] == 1) for i in range(n_classes)])
    if weighted_method=="equally_divided":
        reverse = 1/ls_num_equal1
        total = np.sum(reverse)
        weights = reverse / total
    else:
        weights = ls_num_equal1
    fpr={}
    tpr={}
    auc_list=[]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        auc_list.append(roc_auc_score(labels[:, i], preds[:, i]))
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    # Interpolate all ROC curves at these points
    tpr_ls=[]
    for i in range(n_classes):
        tpr_ls.append(np.interp(fpr_grid, fpr[i], tpr[i]))  # linear interpolation
    weighted_mean_tpr = np.average(tpr_ls, axis=0, weights=weights)
    weighted_auc1 = auc(fpr_grid, weighted_mean_tpr)
    # weighted_auc2 = np.average(auc_list, weights=weights)
    # weighted_auc3 = roc_auc_score(labels, preds, multi_class="ovr", average="weighted",)                        
    return fpr_grid, weighted_mean_tpr, weighted_auc1

def plot_multiclassify_auc_curve(labels, preds, save_path="auc.png", classnames=['class1', 'class2', 'class3']):
    n_classes = labels.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], roc_auc['micro'] = Micro_OvR_AUC(labels, preds)
    fpr["macro"], tpr["macro"], roc_auc['macro'] = Macro_OvR_AUC(labels, preds)
    fpr["weighted"], tpr["weighted"], roc_auc['weighted'] = Weighted_OvR_AUC(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})", color="Gray", linestyle=":", linewidth=4)
    ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})", color="MediumOrchid", linestyle=":", linewidth=4)
    ax.plot(fpr["weighted"], tpr["weighted"], label=f"weighted-average ROC curve (AUC = {roc_auc['weighted']:.2f})", color="Peru", linestyle=":", linewidth=4)
    colors = cycle(['red', 'DeepSkyBlue', 'SeaGreen', 'MediumTurquoise', 'SteelBlue', 'LightPink', 'orange', 'LimeGreen'])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(labels[:, class_id], preds[:, class_id], name=f"ROC curve for {classnames[class_id]}", color=color, ax=ax)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC curves of One-vs-Rest multi-classification")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(prop={"size":8}, frameon=False)
    plt.tight_layout()
    fig.savefig(save_path)
    return roc_auc['micro'], roc_auc['macro'], roc_auc['weighted']

def plot_biclassify_auc_curve(y_true, y_pred, save_path="auc.png", classnames=['negative', 'positive']):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot(fpr, tpr, color='IndianRed', lw=2, label='Overall ROC curve (AUC = %0.2f)' % roc_auc)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(prop={"size":8}, frameon=False)
    plt.tight_layout()
    fig.savefig(save_path)
    return roc_auc

def plot_confuse_matrix(cm, classnames, save_path):
    conf_matrix = pd.DataFrame(cm, index=classnames, columns=classnames)
    fig, ax = plt.subplots(figsize=(4,3), dpi=300)
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15}, cmap="Blues", ax=ax, fmt='d')
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    fig.savefig(save_path)

def plot_regress_curve(y_true, y_pred, pearson_corr, r2, rmse, save_path):
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    data = pd.DataFrame({'True':y_true, 'Predicted':y_pred})
    sns.regplot(x='True', y='Predicted', data=data, scatter_kws={'s': 50}, line_kws={'color': 'red', 'linewidth': 2})
    textstr = '\n'.join((
        f'$R^2={r2:.2f}$',
        f'$RMSE={rmse:.2f}$',
        f'$Pearson\\ Corr={pearson_corr:.2f}$'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('Regression Line with Performance Metrics', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path)


def plot_multiple_regressions(y_true_list, y_pred_list, names, save_path):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    colors = sns.color_palette("hsv", len(y_true_list))
    for i, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        r2 = r2_score(y_true, y_pred)
        data = pd.DataFrame({'True': y_true, 'Predicted': y_pred})
        sns.regplot(x='True', y='Predicted', data=data, scatter_kws={'s': 50, 'color': colors[i]}, 
                    line_kws={'color': colors[i], 'linewidth': 2}, ax=ax)
        ax.text(0.05, 0.95 - i*0.05, f'{names[i]}: $R^2={r2:.2f}$', transform=ax.transAxes, 
                fontsize=12, color=colors[i], verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_title('Multiple Regression Lines with Performance Metrics', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path)

def bi_classify_metrics(labels, preds, plot_cm=False, plot_auc=False, save_path_name='bi', classnames=['Negative','Positive']):
    y_true = de_onehot(labels)
    y_pred = de_onehot(preds)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(int)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0] 
    tp = cm[1][1]
    tpr = rec = sen = round(tp / (tp + fn + 0.0001), 3)
    tnr = spe = round(tn / (tn + fp + 0.0001), 3)
    pre = round(tp / (tp + fp + 0.0001), 3)
    acc = round((tp + tn) / (tp + fp + fn + tn + 0.0001), 3)
    f1 = round((2 * pre * rec) / (pre + rec + 0.0001), 3)
    mcc = matthews_corrcoef(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, preds[:, 1])
    if plot_cm:
        save_path = save_path_name + '_cm.png'
        plot_confuse_matrix(cm, classnames, save_path)
    if plot_auc:
        save_path = save_path_name + '_auc.png'
        plot_biclassify_auc_curve(y_true, preds[:, 1], save_path, classnames)
    metrics = np.array([tn, fp, fn, tp, tpr, tnr, pre, acc, ap, f1, mcc, auc])
    header = ['TN', 'FP', 'FN', 'TP', 'TPR', 'TNR', 'Precision', 'Accuracy', 'AP', 'F1', 'MCC', 'AUC']
    np.savetxt(f'{save_path_name}_metrics.txt', [metrics], delimiter='\t', fmt='%.2f', header='\t'.join(header), comments='')

    return metrics

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true+0.001) )) * 100

def calculate_smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def regression_metrics(y_true, y_pred, plot_regress=True, save_path_name='regress'):
    pearson_corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred)) # mean_squared_error(y_true,y_pred,squared=False)
    evar = explained_variance_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    if plot_regress:
        save_path = save_path_name+'_regress.png'
        plot_regress_curve(y_true, y_pred, pearson_corr, r2, rmse, save_path)
    headers = ['Pearson Correlation', 'R²', 'RMSE', 'MSE', 'Explained Variance', 'MAE', 'MAPE', 'SMAPE']

    np.savetxt(f'{save_path_name}_metrics.txt', np.array([pearson_corr,r2,rmse,mse,evar,mae,mape,smape]).reshape(1, -1), delimiter='\t', fmt='%.2f', header='\t'.join(headers), comments='')
    return 


def metric(job, model_name, cv_taskname, et_taskname=None, classnames=['Non-PE','PE']):
    if job == 'classify':
        result_dir = f'pretrained/{cv_taskname}_{model_name}_{job}'
        for fold in range(1,6):
            if fold == 1:
                preds = np.loadtxt(f'{result_dir}/fold{fold}_val_preds.txt')
                labels = np.loadtxt(f'{result_dir}/fold{fold}_val_labels.txt', dtype=int)
            else:
                preds = np.append(preds, np.loadtxt(f'{result_dir}/fold{fold}_val_preds.txt'))
                labels = np.append(labels, np.loadtxt(result_dir+f'/fold{fold}_val_labels.txt', dtype=int))
        preds = onehot_preds(preds)
        labels = onehot(labels, 2)
        bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name=result_dir+'/cv',classnames=classnames)
        if et_taskname is not None:
            preds = np.loadtxt(f'{result_dir}/test_{et_taskname}_preds.txt')
            labels = np.loadtxt(f'{result_dir}/test_{et_taskname}_labels.txt', dtype=int)
            df = pd.read_csv(f'task/S2814/S2814_processed.csv', sep='\t').iloc[:2689, :]
            df = df.dropna(subset=['SCANEER'])
            preds = onehot_preds(preds)[df.index]
            labels = onehot(labels, 2)[df.index]
            bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name=result_dir+'/test',classnames=classnames)

    else:
        result_dir = f'pretrained/{cv_taskname}_{model_name}_{job}'
        for fold in range(1,6):
            if fold == 1:
                preds = np.loadtxt(result_dir+f'/fold{fold}_val_preds.txt')
                labels = np.loadtxt(result_dir+f'/fold{fold}_val_labels.txt')
            else:
                preds = np.append(preds, np.loadtxt(result_dir+f'/fold{fold}_val_preds.txt'))
                labels = np.append(labels, np.loadtxt(result_dir+f'/fold{fold}_val_labels.txt'))
        regression_metrics(labels, preds, plot_regress=True, save_path_name=result_dir+'/cv')




def metric4EnzyAct(classnames=['Decrease','Increase']):
    df = pd.read_csv(f'task/S2814/S2814_processed.csv', sep='\t')
    preds = df['EnzyAct_CV5_Prob'].values
    labels = df['Label'].values
    preds = onehot_preds(preds)
    labels = onehot(labels, 2)
    bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name='pretrained/EnzyAct/S2814',classnames=classnames)

    df = pd.read_csv(f'task/S10998/S10998_processed.csv', sep='\t')
    preds = df['EnzyAct_CV5_Prob'].values
    labels = df['Label'].values
    preds = onehot_preds(preds)
    labels = onehot(labels, 2)
    bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name='pretrained/EnzyAct/M167',classnames=classnames)

 

def normalize_signed_array(arr):
    arr = np.array(arr, dtype=float)
    preds_norm = np.zeros_like(arr)

    pos_mask = arr > 0
    neg_mask = arr < 0
    zero_mask = arr == 0

    if np.any(pos_mask):
        pos_vals = arr[pos_mask]
        pos_min = pos_vals.min()
        pos_max = pos_vals.max()
        if pos_max != pos_min:
            preds_norm[pos_mask] = 0.5 + 0.5 * (pos_vals - pos_min) / (pos_max - pos_min)
        else:
            preds_norm[pos_mask] = 1.0

    if np.any(neg_mask):
        neg_vals = arr[neg_mask]
        neg_min = neg_vals.min()
        neg_max = neg_vals.max()
        if neg_max != neg_min:
            preds_norm[neg_mask] = 0.5 * (neg_vals - neg_min) / (neg_max - neg_min)
        else:
            preds_norm[neg_mask] = 0.0

    preds_norm[zero_mask] = 0.5
    return preds_norm

def metric4SCANEER(classnames=['Decrease','Increase']):
    df = pd.read_csv(f'task/S2814/S2814_processed.csv', sep='\t')
    df = df.dropna(subset=['SCANEER'])
    preds = df['SCANEER'].values
    preds = normalize_signed_array(preds)
    labels = df['Label'].values
    preds = onehot_preds(preds)
    labels = onehot(labels, 2)
    bi_classify_metrics(labels, preds, plot_cm=True, plot_auc=True, save_path_name='pretrained/SCANEER/S2814',classnames=classnames)




if __name__ == "__main__":
    metric4EnzyAct()
    metric4SCANEER()
    
    metric(job='classify', model_name='GCN', cv_taskname='S10998', et_taskname='S2814', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='GAT', cv_taskname='S10998', et_taskname='S2814', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='MLP', cv_taskname='S10998', et_taskname='S2814', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='RFC', cv_taskname='S10998', et_taskname='S2814', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='SVC', cv_taskname='S10998', et_taskname='S2814', classnames=['Decrease','Increase'])

    metric(job='classify', model_name='MLP', cv_taskname='M576', et_taskname='M167', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='RFC', cv_taskname='M576', et_taskname='M167', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='SVC', cv_taskname='M576', et_taskname='M167', classnames=['Decrease','Increase'])
    metric(job='classify', model_name='MLP', cv_taskname='ProteinGym_clinical', et_taskname=None, classnames=['Pathogenic','Benign'])
    metric(job='classify', model_name='RFC', cv_taskname='ProteinGym_clinical', et_taskname=None, classnames=['Pathogenic','Benign'])
    metric(job='classify', model_name='SVC', cv_taskname='ProteinGym_clinical', et_taskname=None, classnames=['Pathogenic','Benign'])
