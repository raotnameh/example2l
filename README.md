This is a simple example of 2 layer feed forwad ANN.

### Creating the wheel files. 


### Creating the environment to be used. 
conda env create -f environment.yml  <br/>
conda activate oopd  <br/>
python setup.py sdist bdist_wheel
pip install dist/ann-0.0.1-py3-none-any.whl

### Training the example 2 layerd net. 
`
tar -xvf data.tar.gz \
python run.py
`
### Saving the prediction to sql. 
`
.mode csv \
.import data/test_pred_true.csv oopd \
.save data/oopd_pred_true.db
`