jhgcimport pandas as pd;
import numpy as np;
import seaborn as sns;
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def lgb_evaluate(
                learning_rate,
                num_leaves,
                min_split_gain,
                max_depth,
                subsample,
                subsample_freq,
                lambda_l1,
                lambda_l2,
                feature_fraction,
                ):

    clf = lgb.LGBMClassifier(num_leaves              = int(num_leaves),
                             max_depth               = int(max_depth),
                             learning_rate           = 10**learning_rate,
                             n_estimators            = 500,
                             min_split_gain          = min_split_gain,
                             subsample               = subsample,
                             colsample_bytree        = feature_fraction,
                             reg_alpha               = 10**lambda_l1,
                             reg_lambda              = 10**lambda_l2,
                             subsample_freq          = int(subsample_freq),
                             verbose                 = -1
                            )

    global X
    global Y
    
    scores = cross_val_score(clf, X, Y, cv=5, scoring='roc_auc')
    print(np.mean(scores))

    return np.mean(scores)
   
def bayesOpt(train_x, train_y):
    lgbBO = BayesianOptimization(lgb_evaluate, {                                            
                                            'learning_rate':           (-2, 0),
                                            'num_leaves':              (5, 50),
                                            'min_split_gain':          (0, 1),
                                            'max_depth':               (5, 30),
                                            'subsample':               (0.1, 1),
                                            'subsample_freq':          (0, 100),
                                            'lambda_l1':               (-2, 2),
                                            'lambda_l2':               (-2, 2),
                                            'feature_fraction':        (0.1, 1)
                                            })


    lgbBO.maximize(init_points=5, n_iter=5)

    print(lgbBO.res['max'])