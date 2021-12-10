# Run an example. 
from ann import data, example2l
import cProfile,pstats
profiler = cProfile.Profile()
profiler.enable()

print('Reading dataset')
#uncomment to use sql database
# X,Y,X_test,Y_test = data.Mnist(path='data').from_sql('data/oopd_train.db', 'data/oopd_test.db') 
X,Y,X_test,Y_test = data.Mnist(path='data').data()
print("Finished reading the dataset")

print("Training started")
i_d, m_d, o_d = 784, 100, 10
act = 'relu'
lr,epoch = 0.00001,1
net = example2l.Net(i_d,m_d,o_d,act, lr,epoch, X,Y,X_test,Y_test)
net.train(step_u = 64, step=1000)
print("Finished training")


import io
profiler.disable()
s = io.StringIO()
stats = pstats.Stats(profiler,stream=s).sort_stats('ncalls')
stats.print_stats()
with open('test.txt', 'w+') as f:
    f.write(s.getvalue())

