from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')
mnist = input_data.read_data_sets("../data/MNIST/", one_hot=True) 

X_train = mnist.train.images
X_test = mnist.test.images
X_test = X_test - X_train.mean(axis=0);
Y_test = np.argmax(mnist.test.labels,axis=1)

test_image = X_test[478:479]
test_label = Y_test[478]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("SGHMC_MNIST_gt.png")
#plt.show()
plt.close()

480