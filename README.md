# DropOut-SGHMC

## In this repo you can find python code of:
* Logistic Regression
* Stochastic Gradient Hamiltonian Monte Carlo
* DropOut - Stochastic Gradient Hamiltonian Monte Carlo
* Stochastic Gradient Langevin Dinamics
* Utils

## Tested on:

* Adience Age problem
* MNIST Digit Recognition

## Excecution:

unzip datasets

~~~bash

cat data.zip* > data.zip
unzip data.zip

~~~

run scripts

~~~bash

cd models
python <script>.py

~~~


## Requirements:

* Python 2.7

* Tensorflow 1.3 or later, Tensorflow-gpu (alternative)
* Edward 1.3 or later
* Keras 2.1 or later, keras-vggface
* SKlearn, Numpy, Scipy, Pandas, Seaborn, Matplotlib (according to dependence on previous packages)
