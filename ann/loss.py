"""@package docstring
This module contains all the los functions
"""


import numpy as np

class Cross_Ent_Loss:
    """
    Applies the combined cross entropy and softmax loss function element-wise. 
    For easier backprop calculation.

    Args:
        None
    """
    def forward_loss(self,logits,true):
        """
        Args: 
            logits: Predicted vector of shape [num_classes,1].
            true: truth vector of shape [num_classes,1]. 
        """
        self.pred = self.softmax(logits)
        self.true = true
        index = np.where(true == 1)[0]
        loss = true[index] * np.log(self.pred[index])
        return -loss[0][0] # negative sign because to change the gradient direction.
    
    def backward_loss(self):
        """ 
        Implements the backprop calculation.
        """
        dummy = np.zeros_like(self.pred)
        for k,i in enumerate(self.pred):
            dummy[k] = self.pred[k] - self.true[k]
        return dummy
    
    def softmax(self,x):
        """ 
        Implements the softmax function element wise.
        """
        z = np.exp(x)
        self.z = z / sum(z)
        return self.z
