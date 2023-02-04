import numpy as np
import pandas as pd
import tensorflow as tf
from keras import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#region Data Preprocessing

#Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#endregion

#region Building the ANN

#Initializing the ANN
cnn = tf.keras.models.Sequential()

#Convolution
cnn.add(tf.keras.layer.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding a second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Flattening 
cnn.add(tf.keras.layers.Flatten())

#Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#region Training the CNN

#Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

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

#Making a single prediction
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'