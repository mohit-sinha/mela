import pandas as pd
import numpy as np

class MelaClassifier():

	def __init__(self):
		self.var = None
		self.weights = None

	def preprocess(self, data):

		for i in data.columns:
			if data[i].isnull().values().any:
				num = list()
				x = data[i].value_counts().keys()
				num.append(x[0])
				num.append(x[0])
				num.append(x[1])
				data[i] = data[i].fillna(np.random.choice(num))
				

	def probsOf(self, feat, target, train, low, up):
		data = pd.DataFrame()
		data = train.groupby(by=feat)[target].mean()
		for i in data:
			if i > low and i < up:
				i = 0.5
		return data

	def fit(self, train, target):
		self.train = train
		self.target = target
		train_x = train.drop(target, axis=1)
		self.var = train_x.columns
		self.weights = np.ones(self.var.size)		
		self.train_x = train_x

		return self

	def predict(self, test_x):

		X_test = test_x[self.var].copy()
		for feat in self.var:		
			X_test[feat] = test_x[feat].map(self.probsOf(feat, self.target, self.train, 0.45, 0.85))
		X_test.fillna(0.5, inplace=True)

		pred = np.zeros(X_test.shape[0])
		for idx,feat in enumerate(X_test.columns):
			pred += X_test[feat].values*self.weights[idx]
		pred/=sum(self.weights)
		x = np.random.rand(pred.shape[0])
		x/=100
		pred+=x
		return pre
