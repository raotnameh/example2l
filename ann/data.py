"""@package docstring
This module contains the different dataset download functions.
"""

import numpy as np
import requests, gzip, os, hashlib
import sqlite3

#fetch data

class Mnist:
    """
    Download the MNIST dataset and save them in a given path.

    Args:
        path: where to save the dataset.
    
    Return:
        X: training input [60000,784]
        Y: training output [60000,1]

        X_test = test input [10000,784]
        Y_test = test output [10000,1]
    """

    def __init__(self,path):
        self.path = path
    
    def from_sql(self,train_db='oopd_train.db',test_db='oopd_test.db'):
        """
        Load from a given sql database. 
        """
        X, Y = self.sql_database(train_db)
        X_test, Y_test = self.sql_database(test_db)

        return X, Y, X_test, Y_test


    def data(self):
        """
        Download the data from source
        Save it to the disk in self.path/ directory

        """
        X = self.fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
        Y = self.fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:].reshape((-1,1))
        X_test = self.fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
        Y_test = self.fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:].reshape((-1,1))

        rand=np.arange(60000)
        np.random.shuffle(rand)
        X,Y = X[rand], Y[rand]

        return X, Y, X_test, Y_test

    def fetch(self,url):
        """
        Downloads the MNIST data given the url.
        """
        fp = os.path.join(self.path, hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                data = f.read()
        else:
            with open(fp, "wb") as f:
                data = requests.get(url).content
                f.write(data)
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

    def sql_database(self,path_to_db): 
        """
        Read the data from a sql database.
        """
        oopd = sqlite3.connect(path_to_db)
        cursor = oopd.cursor()
        cursor.execute("SELECT * FROM  oopd")
        oopd = cursor.fetchall()

        X,Y = np.zeros((60000,784)), np.zeros((60000,1))
        for k,i in enumerate(oopd):
            x = np.array([i for i in i[:-1]]).reshape(1,-1)
            y = np.array(i[-1][0]).reshape(1,-1)
            X[k+1],Y[k+1] = x,y

        return X[:k], Y[:k]