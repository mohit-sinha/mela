import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MelaClassifier(BaseEstimator):

	def __init__(self, weights, low, up):
		self.var = None
		self.low = low
		self.up = up
		self.weights = np.array(weights)
    

	def preprocess(self, data):
		data = pd.DataFrame(data)
		for i in data.columns:
			if data[i].isnull().values.any:
				x = data[i].value_counts().keys()				
				data[i] = data[i].fillna(x[int(np.random.rand()+0.25)])
		return data
				

	def probsOf(self, feat, train, target):
		data = train.groupby(by=feat)[target].mean()
		for i in data.index:
			if data[i] > self.low and data[i] < self.up:
				data.loc[i] = 0.5
		return data


	def fit(self, train, target):
		self.train = train
		train = self.preprocess(train)
		self.target = target
		train_x = train.drop(target, axis=1)
		self.var = train_x.columns
		if self.var.size > self.weights.size:
			self.weights = np.r_[self.weights, np.zeros(self.var.size-self.weights.size)]
		self.train_x = train_x

		return self


	def predict(self, test_x):
		test_x = self.preprocess(test_x)
		X_test = test_x[self.var].copy()
		for feat in self.var:
			X_test[feat] = test_x[feat].map(self.probsOf(feat, self.train, self.target))

		X_test.fillna(0.5, inplace=True)

		pred = np.zeros(X_test.shape[0])
		for idx,feat in enumerate(X_test.columns):
			pred += X_test[feat].values*self.weights[idx]
		pred/=sum(self.weights)
		noise = np.random.rand(pred.shape[0])
		noise/=100
		pred+=noise

		return pred
