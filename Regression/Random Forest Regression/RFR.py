import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

"""
Random Forest Models are better displayed in 3D not 2D and with multiple features
"""

#Importing the dataset
dataset = pd.read_csv(r'Regression\Random Forest Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1]

#Training the Random Forest Regression model on the whole dataset
#n_estimators = the amount of trees made
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

#Predicting a new result
print(regressor.predict([[6.5]]))

#Visualizing the Random Forest Regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()