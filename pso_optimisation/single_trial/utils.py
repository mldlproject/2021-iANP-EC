# Import libraries
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score

#============================================================
def printPerformance(labels, probs, printout=False, decimal=4):
    predicted_labels = np.round(probs)
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    acc = accuracy_score(labels, predicted_labels)
    ba  = balanced_accuracy_score(labels, predicted_labels)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, predicted_labels)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp)
    f1 = 2*precision*sensitivity / (precision + sensitivity)
    ck = cohen_kappa_score(labels, predicted_labels)
    d = decimal
    if printout:
        print('AUC-ROC: ', round(roc_auc, d))
        print('AUC-PR: ', round(pr_auc, d))
        print('ACC: ', round(acc, d))
        print('BA: ', round(acc, d))
        print('SN/RE: ', round(sensitivity, d))
        print('SP: ', round(specificity, d))
        print('PR: ', round(precision, d))
        print('MCC: ', round(mcc, d))
        print('F1: ', round(f1, d))
    return round(roc_auc, d), round(pr_auc, d), round(acc, d), round(ba, d), round(sensitivity, 4), round(specificity, 4), round(precision, 4), round(mcc, 4), round(f1, 4), round(ck, 4) 

#============================================================
def get_optimasation_function(preds, labels, auc_type='roc'):
    preds_1, preds_2, preds_3, preds_4 = preds[0], preds[1], preds[2], preds[3]
    labels = labels
    #--------------------------------------
    def aucroc_optimisation(weight):
        w1, w2, w3, w4 = weight[0], weight[1], weight[2], weight[3]
        w1_norm = w1/(w1 + w2 + w3 + w4)
        w2_norm = w2/(w1 + w2 + w3 + w4)
        w3_norm = w3/(w1 + w2 + w3 + w4)
        w4_norm = w4/(w1 + w2 + w3 + w4)
        #--------------------------------------
        preds_ensemble = (preds_1*w1_norm + 
                          preds_2*w2_norm + 
                          preds_3*w3_norm + 
                          preds_4*w4_norm)
        #--------------------------------------
        roc_auc = roc_auc_score(labels, preds_ensemble)
        objective = 1 - roc_auc
        return objective
        #--------------------------------------
    def aucpr_optimisation(weight):
        w1, w2, w3, w4 = weight[0], weight[1], weight[2], weight[3]
        w1_norm = w1/(w1 + w2 + w3 + w4)
        w2_norm = w2/(w1 + w2 + w3 + w4)
        w3_norm = w3/(w1 + w2 + w3 + w4)
        w4_norm = w4/(w1 + w2 + w3 + w4)
        #--------------------------------------
        preds_ensemble = (preds_1*w1_norm + 
                          preds_2*w2_norm + 
                          preds_3*w3_norm + 
                          preds_4*w4_norm)
        #--------------------------------------
        roc_auc = average_precision_score(labels, preds_ensemble)
        objective = 1 - roc_auc
        return objective
    if auc_type == 'roc':
        return aucroc_optimisation
    elif auc_type == 'pr':
        return aucpr_optimisation
    
#============================================================
def extract_weight(opt_weight):
    w1, w2, w3, w4 = opt_weight[0], opt_weight[1], opt_weight[2], opt_weight[3]
    w1_norm = w1/(w1 + w2 + w3 + w4)
    w2_norm = w2/(w1 + w2 + w3 + w4)
    w3_norm = w3/(w1 + w2 + w3 + w4)
    w4_norm = w4/(w1 + w2 + w3 + w4)
    return (w1_norm, w2_norm, w3_norm, w4_norm)
