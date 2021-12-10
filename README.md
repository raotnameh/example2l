README  
Tested on Ubuntu 20 and python>=3.8  
This work is done for the CSE600A (Object Oriented Programming and Design) course requirement for group number 41. The problem statement is given in problem.pdf.  

### Overview
1. In this work we implement a simple 2 layer feed-forward arrtificial neural network (ANN) from scratch using numpy only. Gradients are calculated are calculated similarly to this work: http://cs231n.stanford.edu/slides/2021/discussion_2_backprop.pdf.
2. Doxygen file is at latex/refman.pdf 
3. For the profiler file refer to the test/test.txt
4. ann/example2l (2l = 2 layer) class has the implementation of training and testing of the 2 layer network.   
5. We achieve an accuracy of 95 percent when trained for 1 epoch.   
6. Data and output predictions are in data folder (after you erxtract the data.tar.gz).

### Model specifications
```
i_d = 784 # input dimension    
m_d = 100 # no of cells in the mid layer    
o_d = 10 # output dimension    

epoch = 1 # we train for 1 epoch only    
act = 'relu' # activation type      
lr = 0.00001 # learning rate    

X,Y,X_test,Y_test = train_features, train_label, test_features, test_label  
```

### Run these commands to set up the environment. Conda need to be installed. If not than you have to manually install the dependencies. 

```
conda env create -f environment.yml    
conda activate oopd     
python setup.py sdist bdist_wheel      
pip install dist/ann-0.0.1-py3-none-any.whl    
```

### Run these commands to extract the MNIST dataset and to train the example network on it. 
```
tar -xvf data.tar.gz    
```
This will extract the database required to train the network on MNIST dataset. 


``` 
python run.py  
```

After the traning the predicions and their true values will be saved in data/test_pred_true.csv file. To conver this into a sql database run the below commands. 

### Run these commands to save the prediction to a sql database. 
```
.mode csv 
.import data/test_pred_true.csv oopd  
.save data/oopd_pred_true.db   
```
