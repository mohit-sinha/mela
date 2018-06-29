import pandas as pd
import numpy as np
import mela.mela as mela
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def mela_evaluate(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, low, up):
    
    w = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15]
    
    clf = mela.MelaClassifier(w, low, up)  
    target = 'is_pass'
    X = train.drop(target, axis=1)
    y = train[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train[target] = y_train
    clf.fit(X_train, target)
    y_pred = clf.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    
    return score

   
def bayesOpt():
    
    ranges = {                                                
                'w1':  (0.8, 2.5),
                'w2': (0.8, 2.5),
                'w3': (0.8, 2.5),
                'w4': (0.8, 2.5),
                'w5': (0.8, 2.5),                                       
                'w6': (0.8, 2.5),
                'w7':  (0.8, 2.5),
                'w8': (0.8, 2.5),
                'w9': (0.8, 2.5),
                'w10': (0.8, 2.5),
                'w11': (0.8, 2.5),                                       
                'w12': (0.8, 2.5),
                'w13':  (0.8, 2.5),
                'w14': (0.8, 2.5),
                'w15': (0.8, 2.5),
                'low': (0.2, 0.45),
                'up': (0.75, 0.9)
            }
    
    melaBO = BayesianOptimization(mela_evaluate, ranges)


    melaBO.maximize(init_points=50, n_iter=5, kappa = 2, acq = "ei", xi = 0.0)

    bestAUC = round((-1.0 * melaBO.res['max']['max_val']), 6)
    print("\n Best AUC found: %f" % bestAUC)
    print("\n Parameters: %s" % melaBO.res['max']['max_params'])
    

bayesOpt()