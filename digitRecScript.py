import os.path
import gzip
import numpy as np
import matplotlib.pyplot as plt
import keras as kr
import sklearn.preprocessing as pre
import tensorflow as tf
import tkinter as tk # used to load img
from tkinter import filedialog #for uploading image files
from keras.preprocessing import image

def load_data():

    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f: 
         file_content = f.read() 

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        labels = f.read() 

# def neuralNetwork():
from keras.layers import Dense, Dropout, Activation
from keras.models import Model

# Start a neural network, building it by layers.
model = kr.models.Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784, kernel_initializer="normal"))
model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training. 
# They are “dropped-out” randomly. 
# This means that their contribution to the activation of downstream neurons is temporally removed on the 
# forward pass and any weight updates are not applied to the neuron on the backward pass.
model.add(kr.layers.Dense(units=1000, activation='relu'))
model.add(Dropout(0.2))
model.add(kr.layers.Dense(units=1000, activation='relu'))
model.add(Dropout(0.2))
model.add(kr.layers.Dense(units=1000, activation='relu'))


# Add a 10 neuron output layer.
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()
    
train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8)
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)
inputs = train_img.reshape(60000, 784)/255
encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)


if os.path.isfile('savedModel.h5py'): 
    model = kr.models.load_model('savedModel.h5py')
    print("Using precreated neuralNetwork. If you wish to create your own just rename or remove from folder file named savedModel.h5py ")
else:
    model.fit(inputs, outputs, epochs=15, batch_size=100)

# save the current model
kr.models.save_model(
    model,
    "savedModel.h5py",
    overwrite=True,
    include_optimizer=True
)
def testMNIST():
    
    model = kr.models.load_model('savedModel.h5py')
  
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()
        
    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8)
    test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)
    print("MNIST Test Images correctly identified: ",(encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum(),"/10000")

def loadImage():
    root = tk.Tk()
    root.withdraw()
    #code source https://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python
    
    file_path = filedialog.askopenfilename()# opens file select window
    img = image.load_img(path=file_path,color_mode = "grayscale",target_size=(28,28,1))
    #loads image into PIL format
    image1 = np.array(list(image.img_to_array(img))).reshape(1, 784).astype(np.uint8) / 255.0
    # shapes array 
    plt.imshow(img)
    plt.show()
    test = model.predict(image1)
    print("program has predicted : ", test.argmax(axis=1))
    #code source https://towardsdatascience.com/basics-of-image-classification-with-keras-43779a299c8b

def runner():
    opt=True
    while opt:
        print("""          1 Load Image
            2 Run nerual net against test images
            3 Exit """)
        opt= input(" What would you like to do ? ")
        #code source https://stackoverflow.com/questions/19964603/creating-a-menu-in-python

        if opt == "1":
             loadImage()
        elif opt == "2":
             testMNIST()
        elif opt == "3":
            exit()
        else: 
            print("Invalid Entry")

load_data()
#neuralNetwork()
runner()


