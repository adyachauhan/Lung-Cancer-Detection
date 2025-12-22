import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
 
from sklearn.model_selection import train_test_split
from sklearn import metrics
 
import cv2
import gc
import os
 
import tensorflow as tf
from tensorflow import keras
from keras import layers
 
import warnings
warnings.filterwarnings('ignore')
 
import os
 
#DATASET MANAGEMENT/UNDERSTANDING
path = '/Users/adyachauhan/Desktop/abc/xyz'
 
#check if path exists
if not os.path.exists(path):
    print(f"The directory {path} does not exist.")
elif not os.path.isdir(path):
    print(f"The path {path} is not a directory.")
else:
    try:
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        print(classes)
    except NotADirectoryError as e:
        print(f"Error: {e}")
 
for cat in classes:
    image_dir = f'{path}/{cat}'
    images = os.listdir(image_dir)
 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)
 
    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{path}/{cat}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()
 
 
#DATA PREPROCESSING
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64
 
X = []
Y = []
 
for i, cat in enumerate(classes):
    images = glob(f'{path}/{cat}/*.jpeg')
    for image in images:
        img = cv2.imread(image)
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)
 
X = np.asarray(X)
Y = np.asarray(Y)
one_hot_encoded_Y = pd.get_dummies(Y).values
 
X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y,test_size = SPLIT,random_state = 2022)
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
 
 
#MODEL BUILDING
#using a tensorflow library to build the model (keras framework is apt)
#we implement a sequential model with many attributes
model = keras.models.Sequential([
    layers.Conv2D(filters=32,
                kernel_size=(5, 5),
                activation='relu',
                input_shape=(IMG_SIZE,
                            IMG_SIZE,
                            3),
                padding='same'),
    layers.MaxPooling2D(2, 2),
 
    layers.Conv2D(filters=64,
                kernel_size=(3, 3),
                activation='relu',
                padding='same'),
    layers.MaxPooling2D(2, 2),
 
    layers.Conv2D(filters=128,
                kernel_size=(3, 3),
                activation='relu',
                padding='same'),
    layers.MaxPooling2D(2, 2),
 
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])
model.summary()
 
import pydot
 
def check_graphviz():
    #Returns True if Graphviz is available
    try:
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, AttributeError):  # AttributeError added here
        return False
 
keras.utils.plot_model(
    model,
    to_file='model_plot.png',
    show_shapes = True,
    show_dtype = True,
    show_layer_activations = True
)
#3 important parameters when model building
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 
#CALLBACK
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
 
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.90:
            print('\n Validation accuracy has reached upto \90% so, stopping further training.')
            self.model.stop_training = True
 
 
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
 
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
 
#TRAINING THE MODEL
history = model.fit(X_train, Y_train,validation_data = (X_val, Y_val),batch_size = BATCH_SIZE,epochs = EPOCHS,verbose = 1,callbacks = [es, lr, myCallback()])
 
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()
 
#MODEL EVALUATION
Y_pred = model.predict(X_val)
Y_val = np.argmax(Y_val, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)
 
#adding confusion matrix (table used to for performance measurement in classfication)
metrics.confusion_matrix(Y_val, Y_pred)
 
#CLASSIFICATION REPORT
print(metrics.classification_report(Y_val, Y_pred, target_names=classes))

#To save trained model
model.save("models/cnn_model.h5")
print("Model saved to models/cnn_model.h5")
