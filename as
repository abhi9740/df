1.
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
data= pd.read_csv('tennis.csv')
data.head()
X=data.iloc[:,:-1]
print("\nThe First 5 values of the train data is\n", X.head())
y = data.iloc[:, -1]
print("\nThe First 5 values of train output is\n", y.head())
le_outlook=LabelEncoder()
X.Outlook=le_outlook.fit_transform(X.Outlook)
le_Temperature=LabelEncoder()
X.Temperature=le_Temperature.fit_transform(X.Temperature)
le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)
le_Windy = LabelEncoder()
X.Wind = le_Windy.fit_transform(X.Wind)
print("\nNow the Train output is\n", X.head())
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n",y)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
print("Accuracy is:", accuracy_score(classifier.predict(X_test), y_test))
-----------------------------------------------------------------------------------------------------------------------------
2.
import numpy as np
from sklearn import datasets
from sklearn import neighbors
import pylab as pl
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
iris = datasets.load_iris()
print(iris.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 
'data_module'])
n_samples, n_features = iris.data.shape
print((n_samples, n_features))
print(iris.data[0])
print(iris.target.shape)
print(iris.target)
print(iris.target_names)
X, y = iris.data, iris.target
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X, y)
result = clf.predict([[3, 5, 4, 2],])
print(iris.target_names[result])
-----------------------------------------------------------------------------------------------------------------------------
3.
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
X, y =
datasets.make_blobs(n_samples=150,n_features=2,centers=2,cluster_std=1.05,random_state=2)
#Plotting
fig = plt.figure(figsize=(10,8))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')
def step_func(z):
 return 1.0 if (z > 0) else 0.0
def perceptron(X, y, lr, epochs):
 
 # X --> Inputs.
 # y --> labels/target.
 # lr --> learning rate.
 # epochs --> Number of iterations.
 
 # m-> number of training examples
 # n-> number of features 
 m, n = X.shape
 
 # Initializing parapeters(theta) to zeros.
 # +1 in n+1 for the bias term.
 theta = np.zeros((n+1,1))
 
 # Empty list to store how many examples were 
 # misclassified at every iteration.
 n_miss_list = []
 
 # Training.
 for epoch in range(epochs):
 
 # variable to store #misclassified.
 n_miss = 0
 
 # looping for every example.
 for idx, x_i in enumerate(X):
 
 # Insering 1 for bias, X0 = 1.
 x_i = np.insert(x_i, 0, 1).reshape(-1,1)
 
 # Calculating prediction/hypothesis.
 y_hat = step_func(np.dot(x_i.T, theta))
 
 # Updating if the example is misclassified.
 if (np.squeeze(y_hat) - y[idx]) != 0:
 theta += lr*((y[idx] - y_hat)*x_i)
 
 # Incrementing by 1.
 n_miss += 1
 
 # Appending number of misclassified examples
 # at every iteration.
 n_miss_list.append(n_miss)
 
 return theta, n_miss_list
def plot_decision_boundary(X, theta):
 
 # X --> Inputs
 # theta --> parameters
 
 # The Line is y=mx+c
 # So, Equate mx+c = theta0.X0 + theta1.X1 + theta2.X2
 # Solving we find m and c
 x1 = [min(X[:,0]), max(X[:,0])]
 m = -theta[1]/theta[2]
 c = -theta[0]/theta[2]
 x2 = m*x1 + c
 
 # Plotting
 fig = plt.figure(figsize=(10,8))
 plt.plot(X[:, 0][y==0], X[:, 1][y==0], "r^")
 plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
 plt.xlabel("feature 1")
 plt.ylabel("feature 2")
 plt.plot(x1, x2, 'y-')
theta, miss_l = perceptron(X, y, 0.5, 100)
plot_decision_boundary(X, theta)
--------------------------------------------------------------------------------------------------------------------------
4.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
iris_data = load_iris()
X = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = iris_data.target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
clf =MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=5 
,verbose=True, learning_rate_init=0.05)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
fig=ConfusionMatrixDisplay.from_estimator(clf, X_test, 
y_test,display_labels=["Setosa","Versicolor","Virginica"])
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test,y_pred))
----------------------------------------------------------------------------------------------------------------------------
5.
from keras.datasets import mnist
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
#plot the first image in the dataset
plt.imshow(X_train[0])
#check image shape
X_train[0].shape
#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train[0]
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
#predict first 4 images in the test set
model.predict(X_test[:4])
#actual results for first 4 images in test set
y_test[:4]
------------------------------------------------------------------------------------------------------------------
6.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tkinter as tk
from tkinter import filedialog
data_dir = "C:\\Users\\Hello\\Downloads\\archive\\Face Mask Dataset\\Train"
data_dir = pathlib.Path(data_dir)
batch_size = 16
img_height = 64
img_width = 64
rain_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2, 
subset = "training", seed = 123,image_size = (img_height, img_width), batch_size = batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2, subset 
= "validation", seed = 123,image_size = (img_height, img_width), batch_size = batch_size)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)
num_class = 2
model = Sequential([
 layers.experimental.preprocessing.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
 layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(),
 layers.Dropout(0.2),
 layers.Flatten(),
 layers.Dense(64, activation = 'relu'),
 layers.Dense(num_class)
])
noepochs = 7
model.compile(optimizer = 'adam', loss =tf.keras.losses.SparseCategoricalCrossentropy(from_logits =
True),metrics = ['accuracy'])
mymodel = model.fit(train_ds, validation_data = val_ds, epochs = noepochs) #training the model
acc = mymodel.history['accuracy']
val_acc = mymodel.history['val_accuracy']
loss = mymodel.history['loss']
val_loss = mymodel.history['val_loss']
epochs_range = range(noepochs)
plt.figure(figsize=(15, 15)) #creates figure for the plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training loss')
plt.plot(epochs_range, val_loss, label = 'Validation loss')
plt.legend(loc = 'upper right')
plt.title('Training and validation loss')
plt.show()
data_dir = "C:\\Users\\Hello\\Desktop\\img1.JPG"
data_dir = pathlib.Path(data_dir)
batch_size = 16
img_height = 64
img_width = 64
img=keras.preprocessing.image.load_img(data_dir,target_size=(img_height, img_width))
test_image = keras.preprocessing.image.img_to_array(img) # Resize the image
test_image = tf.expand_dims(test_image,0) # Add batch dimension
#test_image = test_image / 255.0 # Normalize pixel values
# Use the trained model to make predictions
prediction = model.predict(test_image)
score=(tf.nn.softmax(prediction[0]))
print("I m here".format(class_names[np.argmax(score)],100*np.max(score)))
if prediction[0][0] >= 0.5:
 print("FaceMask detected!")
else:
 print("No faceMask detected.")
-------------------------------------------------------------------------------------------------------------------------
7.
import torch
from torchvision import models, transforms
from PIL import Image
pip install torchvision
# Load the pretrained DeepLabv3+ model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()
# Preprocess the input image
input_image =Image.open(r"im.jpg")
preprocess = transforms.Compose([
 transforms.Resize((256, 256)),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(input_image).unsqueeze(0)
# Perform semantic segmentation
with torch.no_grad():
 output = model(input_tensor)['out']
output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
# Define the color map for visualization
color_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
 [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
 [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
def grayscale_to_color(grayscale_image):
 """Converts a grayscale image to a color image.
 Args:
 grayscale_image: A 2D NumPy array.
 Returns:
 A 3D NumPy array of the same shape as the input image.
 """
 output = grayscale_image.copy()
 color_map = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
 # Flip the image vertically.
 output = output[::-1, :]
 for i in range(output.shape[0]):
 for j in range(output.shape[1]):
 color = color_map[output[i, j]]
 pixels[j, i] = (color[0], color[1], color[2])
 return pixels
# Save and display the segmented image
segmented_image.save('segmented_image.jpg')
segmented_image.show()
----------------------------------------------------------------------------------------------------------------------------------
