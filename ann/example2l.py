import numpy as np
import pandas as pd
from ann.layer import Linear
from ann.loss import Cross_Ent_Loss
from ann.activation import Relu, Sigmoid

class Net(Linear,Relu,Sigmoid,Cross_Ent_Loss):
    """
    Create a sample 2 layer feed forward artitifical neural network (ANN).

    Args:
        i_d: input dimension.
        m_d: number of cells in 1st layer. 
        o_d: output dimension.
        act: activation function to use.
        lr: Learning rate
        epoch: number of epochs to train. 
        X: training input [60000,784].
        Y: training output [60000,1].
        X_test = test input [10000,784].
        Y_test = test output [10000,1].

    Returns:
        None
    
    Example:
        i_d, m_d, o_d = 784, 100, 10
        act = 'relu'
        lr,epoch = 0.00001,10
        X,Y,X_test,Y_test = data.Mnist(path='data').data()
        net = example2l.Net(i_d,m_d,o_d,act, lr,epoch, X,Y,X_test,Y_test)
        net.train()

    """


    def __init__(self,i_d,m_d,o_d,act, lr,epoch, X,Y,X_test,Y_test):
        
        # Hyper paramneters
        self.alpha = lr
        self.epoch = epoch

        # Layer initialization
        self.layer1 = Linear(i_d,m_d)
        if act == 'relu': 
            self.act1 = Relu()
        else: 
            self.act1 = Sigmoid()
        self.layer2 = Linear(m_d,o_d)

        #Loss initialization
        self.cross_ent_loss = Cross_Ent_Loss()
        
        # Data initialization
        self.X, self.Y = X,Y
        self.X_test, self.Y_test = X_test, Y_test
    
    def forward(self,x): 
        """
        Calls the forward fucntion for each layer. 
        """
        out = self.layer1.forward_l(x)
        out = self.act1.forward_a(out)
        out = self.layer2.forward_l(out)
        
        return out
    
    def backward(self):
        """
        Calls backprop fucntion for each layer to calculate the gradients.
        """
        x = self.cross_ent_loss.backward_loss()
        x = self.layer2.backward_l(x)
        x = self.act1.backward_a(x)
        self.layer1.backward_l(x)
        
    def update(self,):
        """
        Calls the update fucntion of each layer to update the gradients. 
        """
        self.layer2.update_l(self.alpha)
        self.layer1.update_l(self.alpha)
        
    def reset(self,):
        """
        Calls the reset fucntion of each layer to reset the gradients back to 0.
        """
        self.layer2.reset_l()
        self.layer1.reset_l()
        
    def loss(self,pred,index):
        """
        Calculate the cross entropy loss.
        """
        true = np.zeros_like(pred)
        true[int(index[0][0])] = 1
        loss = self.cross_ent_loss.forward_loss(pred,true)
        return loss
    
    def evaluate(self,X=None,Y=None,save=True):
        """
        Evaulate the created model given the test data. 
        """
        if X: 
            self.X_test = X
            self.Y_test = Y
        pred = 0
        if save: dummy_pred, dummy_true = [], []
        for k,_ in enumerate(self.X_test):
            x, y = self.X_test[k].reshape(-1,1), self.Y_test[k][0]
            logits = self.forward(x)
            if np.argmax(logits) == y:
                pred += 1
            
            if save: 
                dummy_pred.append(np.argmax(logits))
                dummy_true.append(y)

        print(f"Accuracy of the model is: {pred/k}")

        if save: 
            out = pd.DataFrame(list(zip(dummy_true, dummy_pred)),columns =['True', 'Pred'])
            out.to_csv('data/test_pred_true.csv', encoding='utf-8', index=False)
                
    def train(self,step_u=1,step=100000,test=False):
        """
        Train the created model given the input.
        """
        self.reset()
        for ep in range(self.epoch):
            dummy,dummy_e = 0,0
            for k,_ in enumerate(self.X):
                x, y = self.X[k].reshape(-1,1), self.Y[k].reshape(-1,1)
                logits = self.forward(x)
                loss = self.loss(logits,y)
                dummy += loss
                dummy_e += loss

                self.backward()
                
                if (k+1) % step_u == 0:
                    self.update()
                    
                if (k+1) %step == 0: 
                    print(f"Loss after epoch({ep+1})-iteration({k+1}/{len(self.X)}) is \t\t :{dummy/step}")
                    dummy = 0
                if test: break

            self.alpha *= 0.5
            print(f"Loss after epoch {ep+1} is :{dummy_e/k}")
            print(f"Learning rate after epoch {ep+1} is :{self.alpha}")        
            self.evaluate()
            dummy_e = 0

