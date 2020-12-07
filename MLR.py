import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("F:/AITutorials/projects/Uber_Prediction/taxi.csv")
data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

reg =LinearRegression()

reg.fit(x_train, y_train)
#print("training score: ", reg.score(x_train, y_train))
#print("test score: ", reg.score(x_test, y_test))

pickle.dump(reg, open('taxi.pkl', 'wb'))
model = pickle.load(open('taxi.pkl', 'rb'))

print(model.predict([[80, 1770000, 6000, 85]]))


