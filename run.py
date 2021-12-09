# Run an example. 
from ann import data, example2l

i_d, m_d, o_d = 784, 100, 10
act = 'relu'
lr,epoch = 0.00001,1
X,Y,X_test,Y_test = data.Mnist(path='data').data()
net = example2l.Net(i_d,m_d,o_d,act, lr,epoch, X,Y,X_test,Y_test)
net.train(step_u = 128, step=1000)