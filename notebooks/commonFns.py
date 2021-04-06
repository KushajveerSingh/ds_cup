from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score, recall_score,precision_score,roc_curve,roc_auc_score,RocCurveDisplay,f1_score
smallestF =  np.finfo('float').eps # smallest float value (numpy), will be used later
def print_size(df, name):
    counts = df['Default_ind'].value_counts()
    print(f'{name} dataset')
    print(f'Num 0 values = {counts[0]}')
    print(f'Num 1 values = {counts[1]}')
    print()
def removenulls(DF):
    DF['uti_card_50plus_pct'].fillna(DF['uti_card_50plus_pct'].median(),inplace=True)
    DF['rep_income'].fillna(DF['rep_income'].median(),inplace=True)
    return DF
def oneHotEncodeStates(DF):
    _col = 'States'
    DF_Fixed = DF.copy()
    enc = OneHotEncoder(handle_unknown='ignore',sparse = False)
    SM = enc.fit_transform(DF[_col].to_numpy().reshape(-1,1)) #Got it!
    for ind,cat in zip(range(0,len(enc.categories_[0])),enc.categories_[0]):
        DF_Fixed.drop(columns=['is'+str(cat)],inplace=True,errors='ignore')
        DF_Fixed.insert(0,'is'+str(cat), SM[:,ind]) # Inserted column names are isAK, isAL, isDC..etc. 
    #And it will contain 0 if that is not the state and 1 if that is the state,
    DF_Fixed.drop(columns=[_col],inplace=True) #Remove States Column
    return DF_Fixed
def resampleMinority(DF):
    DF = DF.append(DF[DF.Default_ind == 1])
    DF = DF.append(DF[DF.Default_ind == 1])
    DF = DF.append(DF[DF.Default_ind == 1])
    DF = DF.append(DF[DF.Default_ind == 1])
    return DF
def getMetrics(Ytrue,Ypred,YpredProb,isPrint = True):
    accScore = accuracy_score(Ytrue,Ypred)
    recScore = recall_score(Ytrue,Ypred)
    preScore = precision_score(Ytrue,Ypred)
    confMatrix = pd.crosstab(pd.Series(Ytrue.values,name='Actual'),pd.Series(Ypred,name='Predicted'))
    f1 = f1_score(Ytrue,Ypred)
    fpr, tpr, thresholds = roc_curve(Ytrue, YpredProb, pos_label=1)
    auc = roc_auc_score(Ytrue,YpredProb)
    if(isPrint):
        print('Accuracy =',accScore)
        print('Recall =',recScore)
        print('Precision =',preScore)
        print('--------------------------')
        print('Confusion Matrix')
        print(confMatrix)
        print('--------------------------')
        print('F1 Score',f1)
        print('Roc Auc Score = ',auc)
        #print(fpr,tpr)
        display = RocCurveDisplay(fpr=fpr,tpr=tpr)
        display.plot()
        #plt.show()
    return {'Accuracy':accScore,'Recall':recScore,'Precision':preScore,
    'ConfMatrix':confMatrix,'F1':f1,'FPR':fpr,'TPR':tpr,'AUC': auc,
    'Thresholds':thresholds}

def scaleColumn(col):
    sc = StandardScaler()
    col = sc.fit_transform(col)
    return col

def createThreshDict(colName,Arr):
    _dict = {}
    _dict['colName'] = colName
    _dict['ThreshArr'] = Arr
    return _dict

def binColumn(col,threshArr=None):
    assert col.isnull().all() != True, "All values are null?"
    def binVal(val):
        loc = 1
        for x in threshArr:
            if(val<=x):
                return loc
            else:
                loc = loc+1
        return loc
    theCol = col.copy()
    if(threshArr is None):
        threshArr = col.quantile([0.2,0.4,0.6,0.8]).to_numpy()
    threshArr = np.sort(np.array(threshArr))
    #Replace Nulls with 0 if any
    theBinnedCol = []
    for val in theCol:
        if(not np.isnan(val)):
            theBinnedCol.extend([binVal(val)])
        else:
            theBinnedCol.extend([0])
        #theCol.update(pd.Series([], index=[ind]))
    return pd.Series(theBinnedCol,index=col.keys())

def dropColsNotInList(DF,colList):
    return DF.drop(columns=[col for col in DF if col not in colList])
