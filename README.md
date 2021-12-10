This is a simple example of 2 layer feed forwad ANN.

### Creating the wheel files. 
1. python setup.py sdist bdist_wheel

### Creating the environment to be used. 
`
conda env create -f environment.yml 

conda activate oopd 

pip install dist/ann-0.0.1-py3-none-any.whl
`

### Training the example 2 layerd net. 
1. tar -xvf data.tar.gz 
2. python run.py

### Saving the prediction to sql. 
1. .mode csv
2. .import data/test_pred_true.csv oopd
3. .save data/oopd_pred_true.db
