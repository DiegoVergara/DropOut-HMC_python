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



## Requirements:

* Python 2.7

* Tensorflow 1.3 or later, Tensorflow-gpu (alternative)
* Edward 1.3 or later
* Keras 2.1 or later, keras-vggface
* SKlearn, Numpy, Scipy, Pandas, Seaborn, Matplotlib (according to dependence on previous packages)
