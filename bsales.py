import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import linear_model


#feature engineering
def make_features(filename):
#	df.info()#this would give info on the dataframe
	#df.Item_Weight.value_counts()	
	#df.dropna(how='any').shape
#df = pd.read_csv('Train.csv')
	df = pd.read_csv(filename)
	#filling null values-either drop, or fill with mean, but what about string types
	df.fillna(method='ffill', inplace=True)
	#coding categorical variables
	df.drop(['Item_Identifier', 'Outlet_Identifier'], axis = 1, inplace=True)
	df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Establishment_Year'], drop_first=True)
	return df

#add same features to the training and testing data
train = make_features('Train.csv')
y = train.Item_Outlet_Sales
X_train = train.drop('Item_Outlet_Sales', axis = 1)
test = make_features('Test.csv')
regr = linear_model.LinearRegression()
regr.fit(X_train, y)
#df.head()

test_sales = regr.predict(test)

#submission
submission = pd.read_csv('SampleSubmission.csv')
#the follwoing are deleted from test dataframe!!!!
#submission.Item_Identifier = test.Item_Identifier
#submission.Outlet_Identifier = test.Outlet_Identifier
submission.Item_Outlet_Sales = test_sales
submission.to_csv('submission1.csv', index = False)

