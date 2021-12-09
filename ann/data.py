import numpy as np
import requests, gzip, os, hashlib

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
    
    def data(self):
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

