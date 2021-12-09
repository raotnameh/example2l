This is a simple example of 2 layer feed forwad ANN.

### Creating the wheel files. 

1. Run python setup.py sdist bdist_wheel
2. pip install dist/ann-0.0.1-py3-none-any.whl"

### Creating the environment to be used. 

1. conda env create -f environment.yml
2. conda activate oopd

### Training the example 2 layerd net. 

1. create a folder named data/
2. python run.py