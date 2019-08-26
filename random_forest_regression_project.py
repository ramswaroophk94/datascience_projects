# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('BlackFriday_clean.csv')
#X = dataset.iloc[:, :-1].values
X = pd.DataFrame(dataset.iloc[:, :-1].values)
y = dataset.iloc[:, 11].values

X = X.drop(X.columns[[0, 1]], axis=1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoderAge = LabelEncoder()
X.iloc[:,1] = labelEncoderAge.fit_transform(X.iloc[:,1])

#Encoding gender
labelEncoderGender = LabelEncoder()
X.iloc[:,0] = labelEncoderGender.fit_transform(X.iloc[:,0])

#Encoding City category
labelEncoderCity = LabelEncoder()
X.iloc[:,3] = labelEncoderCity.fit_transform(X.iloc[:,3])

#Encoding current city yr
labelEncoderCityyr = LabelEncoder()
X.iloc[:,4] = labelEncoderCityyr.fit_transform(X.iloc[:,4])




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Visualising the Random Forest Regression results (higher resolution)
'''X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Black Friday Purchase (Random Forest Regression)')
plt.xlabel('')
plt.ylabel('')
plt.show() '''

#Calculating the R-Squared

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)  

#calculating the RMSE 

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Visualizing the Random Forest Regression

from  matplotlib import pyplot 
pyplot.scatter(y_test, y_pred)
pyplot.plot(y_test,y_pred)
pyplot.show()







