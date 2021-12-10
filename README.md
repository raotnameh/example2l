##### Tesdted on Ubuntu 20 and python>=3.8
##### This project is done for the CSE600A (Object Oriented Programming and Design) course requirement.
### Overview
1. In this work we implement a simple 2 layer feed forwad arrtificial neural netowork (ANN) from scratch using numpy only. Gradients are calculated locally and passed for downstream calculations similarly to this work: http://cs231n.stanford.edu/slides/2021/discussion_2_backprop.pdf.
2. For the hierchy of different classes refer to the latex/refman.pdf file. 
3. For the profiler file refer to the test/test.txt

#### Model specifications
```
i_d = 784 # input dimension  
m_d = 100 # no of cells in the mid layer  
o_d = 10 # output dimension  

epoch = 1 # we trainn for 1 epoch only  
act = 'relu' # activation type  
lr = 0.00001 # learning rate  

X,Y,X_test,Y_test = train_features, train_label, test_features, test_label  
```
1. example2l (2l > 2 layer) class has implements the training and testting of the 2 layer network.   
2. We achieve an accuracy of 95 percent with these hyperparameters.   


### Run these commands to set up the environment. We need conda to be installed. If not than you have to manually install the dependencies. 
```
conda env create -f environment.yml  <br/>
conda activate oopd  <br/>
python setup.py sdist bdist_wheel  <br/>
pip install dist/ann-0.0.1-py3-none-any.whl
```
### Run these commands to extract the MNIST dataset and to train the example network on it. 
```
tar -xvf data.tar.gz <br/>
python run.py
```

### Run these commands to save the prediction to a sql database. 
```
.mode csv <br/>
.import data/test_pred_true.csv oopd <br/>
.save data/oopd_pred_true.db 
```

