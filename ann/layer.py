import numpy as np

class Linear:
    """
    This class implements the linear layer.
    
    Args: 
        i_d > input dimension
        o_d > output dimension

    Attributes:
        weight: the leranable weight of the module.
        bias: the leranable bias of the module.

        weight_grad: gradient for the weight matrix.
        bias_grad: gradient for the bias matrix.

    Examples:
     >>>> m = ann.Linear(10,20)
     >>>> inp = np.random.rand(i_d,1)
     >>>> out = m.forward(inp)
     >>>> out.shape
     [o_d,1]

    """

    def __init__(self,i_d,o_d):
        # Inititalizing weight and bias matrices using numpy
        self.weight = (np.random.rand(o_d,i_d)-0.5)/np.sqrt(i_d) 
        self.bias = np.zeros((o_d,1))

        # Initializing weight and bias gradient counterparts. 
        self.weight_grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)

    def forward_l(self,x):
        '''
        Implements the forward calculation given an input. 

        Input: matrix with shape [dim,1]
        ouput: matrix with shape [o_d,1]
        ''' 

        # Saving the input for backward calculation. 
        self.x = x 
        return np.dot(self.weight,x) #+ self.bias
    
    def backward_l(self,x):
        ''' 
        Implements the  Backprop calculations given the gradeints.
        '''
        self.weight_grad += np.dot(x,self.x.T)
        return np.dot(self.weight.T,x)
       
    def update_l(self,alpha):
        """
        To update the Gradients of each layer.
        """
        self.weight = self.weight - alpha*self.weight_grad       
        self.reset_l()

    def reset_l(self,):
        """
        To reset th gradients to zero.
        """
        self.weight_grad = np.zeros_like(self.weight)