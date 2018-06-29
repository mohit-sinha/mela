import pandas as pd
import numpy as np
import mela.mela as mela
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def mela_evaluate(                
                w1,
                w2,
                w3,
                w4,
                w5,
                w6,
                w7,
                w8,
                w9,
                w10,
                w11,
                w12,
                w13,
                w14,
                w15,
                low,
                up
                ):
    
    w = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15]
    
    clf = mela.MelaClassifier(w, low, up)    
    scores = cross_val_score(clf, train_x, train_y, cv=5, scoring='roc_auc')
    print(np.mean(scores))

    return np.mean(scores)
   
def bayesOpt():
    w = np.random.rand(train.shape[1] - 1)*2
    melaBO = BayesianOptimization(mela_evaluate, {                                                
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
                                            })


    melaBO.maximize(init_points=5, n_iter=5)

    print(melaBO.res['max'])
    

bayesOpt()