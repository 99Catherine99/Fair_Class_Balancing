"""
Utility functions for model performance evaluation

"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
import itertools  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def performance(df_pred, vis = False):

    y_test = df_pred['truth']
    y_pre = df_pred['pred']
    
    print("Test Accuracy ",accuracy_score(y_test,y_pre))
    print ("Precision: %f"%precision_score(y_test,y_pre,average='weighted'))
    print ("Recall: %f"%recall_score(y_test,y_pre,average='weighted'))
    print ("F1: %f"%f1_score(y_test,y_pre,average='weighted'))

    if(vis):
        classes=np.arange(2)
        cm = confusion_matrix(y_test,y_pre) 
        cm_n=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tnr, fpr, fnr, tpr=cm_n.ravel()
        print("Normalized confusion matrix")
        print(cm)
        
        print('Confusion matrix, without normalization')
        print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        
        plt.figure(figsize=(12,4))
        # Plot the confusion matrix
        plt.subplot(1,2,1)
        plot_confusion_matrix(cm, classes=classes,
                              title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.subplot(1,2,2)
        plot_confusion_matrix(cm, classes=classes, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()

def group_comp(df_pred,label,privileged_group):
    ### 
    g1 = privileged_group 
    g0 = privileged_group^1

    #privileged group
    df_male=df_pred[df_pred[label]==g1]
    male_truth=df_pred[df_pred[label]==g1]['truth']
    male_pred=df_pred[df_pred[label]==g1]['pred']

    pr1=len([i for i in male_pred if i==1])/len(male_pred)
    classes=np.arange(2)
    cm = confusion_matrix(male_truth,male_pred) 
    tn1, fp1, fn1, tp1=cm.ravel()
    g1_results = [ f1_score(male_truth,male_pred,average='weighted'), tp1/(tp1+fn1), fp1/(fp1+tn1), pr1]
    
    
    # non-privileged group
    df_female=df_pred[df_pred[label]==g0]
    female_truth=df_pred[df_pred[label]==g0]['truth']
    female_pred=df_pred[df_pred[label]==g0]['pred']
    
    pr0=len([i for i in female_pred if i==1])/len(female_pred)
    classes=np.arange(2)
    cm = confusion_matrix(female_truth,female_pred) 
    tn0, fp0, fn0, tp0=cm.ravel()
    
    g0_results = [f1_score(female_truth,female_pred,average='weighted'), tp0/(tp0+fn0), fp0/(fp0+tn0), pr0]

    
    ##print the summary of comprision
    table = [['Group', 'F1', 'TPR', 'FPR', 'PR'], ['Privileged']+g1_results, ['Non-privileged']+g0_results]
    print(tabulate(table, floatfmt='.3f', headers = "firstrow", tablefmt='psql'))

    eop=tp0/(tp0+fn0)-tp1/(tp1+fn1)
    eodds=(tp0/(tp0+fn0)-tp1/(tp1+fn1))*0.5+(fp0/(fp0+tn0)-fp1/(fp1+tn1))*0.5
    sp=pr0-pr1
    print("Equal Opportunity %.3f"%(tp0/(tp0+fn0)-tp1/(tp1+fn1)))
    print("Equal Odds %.3f" %((tp0/(tp0+fn0)-tp1/(tp1+fn1))*0.5+(fp0/(fp0+tn0)-fp1/(fp1+tn1))*0.5))
    print("Statistical Parity %.3f"%(pr0-pr1))
    
    
    return eop, eodds, sp