import pandas as pd
import numpy as np

class MelaClassifier():

	def __init__(self):
		self.var = None
		

	def preprocess(self, data):

		for i in data.columns:
			if data[i].isnull().values().any:
				x = data[i].value_counts().keys()				
				data[i] = data[i].fillna(x[int(np.random.rand()+0.25)])
				

	def probsOf(self, feat, train, target):
		data = pd.DataFrame()
		data = train.groupby(by=feat)[target].mean()
		for i in data:
			if i > self.low and i < self.up:
				i = 0.5
		return data

	def fit(self, train, target):
		self.train = train
		train = self.preprocess(train)
		self.target = target
		train_x = train.drop(target, axis=1)
		self.var = train_x.columns
			
		self.train_x = train_x

		return self

	def predict(self, test_x):
		test_x = self.preprocess(test_x)
		X_test = test_x[self.var].copy()
		for feat in self.var:		
			X_test[feat] = test_x[feat].map(self.probsOf(feat, self.target, self.train, 0.45, 0.85))
		X_test.fillna(0.5, inplace=True)

		pred = np.zeros(X_test.shape[0])
		for idx,feat in enumerate(X_test.columns):
			pred += X_test[feat].values*self.weights[idx]
		pred/=sum(self.weights)
		noise = np.random.rand(pred.shape[0])
		noise/=100
		pred+=noise

		return pred
