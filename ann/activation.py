"""@package docstring
This module contains the different activation functions used in AI. 
"""

import numpy as np

class Relu:
    """
    Applies the rectified linear unit function element-wise.

    Args:
        None
    """

    def forward_a(self,x):
        """
        Implements the Forward calculation given an input.
        """
        self.z = np.maximum(0,x)
        return self.z
    
    def backward_a(self,x):
        """ 
        Implements the backprop calculation.
        """
        dummy = np.zeros_like(self.z)
        for k,i in enumerate(self.z):
            if i > 0: dummy[k] = x[k]
            else: dummy[k] = 0.0
        return dummy    
    
class Sigmoid:
    """
    Applies the sigmoid function element-wise.

    Args:
        None
    """
    def forward_a(self,x):
        """
        Implements the Forward calculation given an input.
        """
        z = np.exp(-x)
        self.z = 1 / (1 + z)
        return self.z
    
    def backward_a(self,x):
        """ 
        Implements the backprop calculation.
        """
        return (self.z * (1-self.z)) * x
    