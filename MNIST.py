import os.path
import gzip
import numpy as np
import matplotlib.pyplot as plt
import keras as kr
import sklearn.preprocessing as pre
import tensorflow as tf

def load_data():

    print("          Start IMAGES FILE IMPORTED")
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f: 
         file_content = f.read() 
    print("          IMAGES FILE IMPORTED")

    type(file_content)
    print(type(file_content)) 
    print(file_content[0:4]) 
    print(int.from_bytes(file_content[0:4], byteorder='big')) 
    print(int.from_bytes(file_content[8:12], byteorder='big'))

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        labels = f.read() 

    print("          LABELS FILE IMPORTED")
    myInt = int.from_bytes(labels[4:8], byteorder="big") 

    print(myInt) 

    myInt = int.from_bytes(labels[8:9], byteorder="big") 
    print(myInt) 


    l = file_content[16:800]
    type(l)
    

    image = ~np.array(list(file_content[16:800])).reshape(28,28).astype(np.uint8)

def neuralNetwork():
    from keras.layers import Dense, Dropout, Activation
    from keras.models import Model

    # Start a neural network, building it by layers.
    model = kr.models.Sequential()

    # Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=1000, activation='relu', input_dim=784))
    model.add(Dropout(0.2)) #Dropout is a technique where randomly selected neurons are ignored during training. 
    # They are “dropped-out” randomly. 
    # This means that their contribution to the activation of downstream neurons is temporally removed on the 
    # forward pass and any weight updates are not applied to the neuron on the backward pass.
    model.add(kr.layers.Dense(units=1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(kr.layers.Dense(units=1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(kr.layers.Dense(units=1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(kr.layers.Dense(units=1000, activation='relu'))

    # Add a 10 neuron output layer.
    model.add(kr.layers.Dense(units=10, activation='softmax'))

    # Build the graph.
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
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


    print(train_lbl[0], outputs[0])

    if os.path.isfile('savedModel.h5py'): 
        model = kr.models.load_model('savedModel.h5py')
    else:
        model.fit(inputs, outputs, epochs=15, batch_size=1000)

    # save the current model
    kr.models.save_model(
        model,
        "savedModel.h5py",
        overwrite=True,
        include_optimizer=True
    )
   
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read()
        
    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8)
    test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)
    print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum())

load_data()
neuralNetwork()


