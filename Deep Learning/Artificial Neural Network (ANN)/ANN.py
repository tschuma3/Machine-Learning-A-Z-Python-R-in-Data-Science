import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#region Data Preprocessing

#Importing the dataset
dataset = pd.read_csv(r'Deep Learning\Artificial Neural Network (ANN)\Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#Label Encoding the "Gender" column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#One Hot Encoding the "Geography" column
ct = ColumnTransformer(transforms=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#endregion

#region Building the ANN

#Initializing the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layer.Dense(units=6, activation='relu'))

#Adding the second hidden layer
ann.add(tf.keras.layer.Dense(units=6, activation='relu'))

#Adding the output layer
ann.add(tf.keras.layer.Dense(units=1, activation='sigmoid'))

#region Training the ANN

#Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#endregion

#region Making the prediction and evaluating the model

#Predicting the result of a single observation
print(ann.predict(sc.transform[[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]) > 0.5)

#Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

#endregion