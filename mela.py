import pandas as pd
import numpy as np

def probsOf(feat, target):
    data = pd.DataFrame()
    data = train.groupby(by=feat)[target].mean()
    for i in data:
        if i > 0.4 and i < 0.85:
            i = 0.5
    return data

def train(train, test, target):
	var = ['trainee_id', 'test_id', 'difficulty_level', 'trainee_engagement_rating', 'total_programs_enrolled', 'age']
	weights = [1, 1.7, 1.4, 1.1, 1, 1]
	X_train = train[var].copy()
	X_test = test[var].copy()

	for feat in X_train.columns:
		X_train[feat] = train[feat].map(probsOf(feat, target)) #train.groupby(by=feat)['is_pass'].mean()
    	X_test[feat] = test[feat].map(probsOf(feat, target)) #train.groupby(by=feat)['is_pass'].mean()
	X_test.fillna(0.5, inplace=True)

	pred = np.zeros(X_test.shape[0])
	for idx,feat in enumerate(X_test.columns):
		pred += X_test[feat].values*weights[idx]
	pred/=sum(weights)
	x = np.random.rand(pred.shape[0])
	x/=100
	pred+=x

	return pred

sub = train(train, test, 'is_pass')