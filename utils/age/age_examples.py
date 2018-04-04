import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from PIL import Image
plt.switch_backend('agg')
test_data_path = genfromtxt("age_test.txt", delimiter=" ", dtype =None)

ix = 76

path = test_data_path[ix-2]

img = Image.open('data/'+path[0])


#pixels = img.reshape((28, 28))
size = 224, 224
img.thumbnail(size,Image.ANTIALIAS)
plt.imshow(img)
plt.savefig("SGHMC_ADIENCE_gt_"+str(ix)+".png")
#plt.show()
plt.close()