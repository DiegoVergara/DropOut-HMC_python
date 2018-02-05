from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
import numpy as np
from numpy import genfromtxt
from keras.preprocessing import image
from keras_vggface import utils

train_data_path = genfromtxt("../data/ADIENCE/age_train.txt", delimiter=" ", dtype =None, encoding ='utf-8')
test_data_path = genfromtxt("../data/ADIENCE/age_test.txt", delimiter=" ", dtype =None, encoding ='utf-8')

# Convolution Features
vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

# After this point you can use your model to predict.

print("Train Data")
f = open('../data/ADIENCE/vgg_face/X_train.csv', 'ab')
f2 = open('../data/ADIENCE/vgg_face/Y_train.csv', 'ab')


i = 0
for path, c in train_data_path:
	print(i)
	i = i+1
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = utils.preprocess_input(x, version=1) # or version
	features = vgg_features.predict(x)
	np.savetxt(f, features, fmt="%1.8f", delimiter=', ')
	np.savetxt(f2, np.array([c]), fmt="%d", delimiter=', ')
f.close()
f2.close()

print("Test Data")
f3 = open('../data/ADIENCE/vgg_face/X_test.csv', 'ab')
f4 = open('../data/ADIENCE/vgg_face/Y_test.csv', 'ab')

i = 0
for path, c in test_data_path:
	print(i)
	i = i+1
	img = image.load_img(path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = utils.preprocess_input(x, version=1) # or version
	features = vgg_features.predict(x)
	np.savetxt(f, features, fmt="%1.8f", delimiter=', ')
	np.savetxt(f2, np.array([c]), fmt="%d", delimiter=', ')
f3.close()
f4.close()