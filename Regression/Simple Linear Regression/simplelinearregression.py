import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
Equation: yhat = b0 + b1X1 <-- Represents a sloped line
            y = mx + b

Assumptions of Linear Regression
1. Linearity (Linear relationship between Y and each X)
2. Homoscedasticity (Equal Variance)
3. Multivariate Normality (Normality of error distribution)
4. Independence (of observations. Includes "no autocorrelation")
5. Lack of Multicollinearity (Predictors (X) are not correlated with each other)
6. The Outlier Check (This is not an assumption, but an "extra")
"""


#Importing the dataset
dataset = pd.read_csv(r'Regression\Simple Linear Regression\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #Would get the same regression line if both X_trains were switched for X_test
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()