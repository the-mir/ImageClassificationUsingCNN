import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

DIRECTORY = r"C:\Users\88016\Downloads\CNN\Dataset\leaf"
CATEGORIES = ['Strawberry_fresh', 'Strawberry_scrotch']

data = []



for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (100, 100))
        data.append([img_arr, label])
#print(data)
random.shuffle(data)

x = []
y = []

for features, label in data:
    x.append(features)
    y.append(label)

X = np.array(x)
Y = np.array(y)
X = X / 255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=15, validation_split=0.1)

# Split the data into training and testing sets
split = int(0.9 * len(X))  # 90% for training, 10% for testing
x_train, x_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Generate predictions on the test set
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Generate classification report
report = classification_report(y_test, y_pred, target_names=CATEGORIES)
print(report)

# Model Prediction
img_path = r"C:\Users\88016\Downloads\CNN\Dataset\leaf\Strawberry_fresh\0e527d62-de6c-4ab5-bb49-f7a6686d207b___RS_HL 4808.JPG"
img = cv2.imread(img_path)
img = cv2.resize(img, (100, 100))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Perform prediction
prediction = model.predict(img)
class_index = np.argmax(prediction)
class_label = CATEGORIES[class_index]
accuracy = prediction[0][class_index] * 100

print("Prediction:", class_label," || Accuracy:", accuracy, "%")