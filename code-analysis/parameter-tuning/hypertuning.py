# hypertuning of parameters (xgboost)

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


x_train = pd.read_csv('x_train.csv')
target = 'Disbursed'
IDcol = 'ID'




# function to do cross validation 
def modelfit(alg, dtrain, y_train, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y_train ,eval_metric='logloss')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.log_loss(y_train, dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')




################################################################################
# 1.Fix learning rate and number of estimators for tuning tree-based parameters 
################################################################################

#Choose all predictors except target & IDcols
predictors = [x for x in x_train.columns]
xgb1 = XGBClassifier(
 learning_rate =0.15,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=random)
modelfit(xgb1, x_train, y_train, predictors)




################################################################################
################ 2.Tune max_depth and min_child_weight########################## 
################################################################################


param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
} 

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.15, n_estimators=500, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=random), 
 param_grid = param_test1, scoring='logloss',n_jobs=4,iid=False, cv=5)



