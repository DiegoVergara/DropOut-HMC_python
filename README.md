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


## Keras VGG_Face Age Dataset Creation:

For the creation of the features through Keras VGG_Face, first it is necessary to download the ADIENCE faces database from the following links:

https://drive.google.com/drive/folders/1A0EDo0oYH3pBEZyq6zfk_jVg8ZvYM2cE?usp=sharing

or

https://www.openu.ac.il/home/hassner/Adience/data.html

Download "aligned.tar.gz" archive, then:

~~~bash

mv download_path/aligned.tar.gz dropout-hmc_python/data/

cd dropout-hmc_python/data/

tar -xvf aligned.tar.gz 

~~~


Run python scripts:

~~~bash

cd dropout-hmc_python/utils/

python keras_vgg_face_features.py

~~~
