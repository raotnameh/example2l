This is a simple example of 2 layer feed forwad ANN.

## Sample run

1. create a folder named data/
2. pip install dist/ann-0.0.1-py3-none-any.whl"

Run these commands 

```
i_d, m_d, o_d = 784, 100, 10
act = 'relu'
lr,epoch = 0.00001,10
X,Y,X_test,Y_test = data.Mnist(path='data').data()
net = example2l.Net(i_d,m_d,o_d,act, lr,epoch, X,Y,X_test,Y_test)
net.train(step_u = 128, step=1000)
```