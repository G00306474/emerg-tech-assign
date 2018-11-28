import gzip
import numpy as np
import tensorflow as tf
import numpy as np
import keras as kr
import sklearn.preprocessing as pre

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

load_data()

