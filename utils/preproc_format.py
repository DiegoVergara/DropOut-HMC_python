import numpy as np
import pandas as pd
from numpy import genfromtxt
all_dir = genfromtxt("../data/ADIENCE/dataset.txt", delimiter = ",", dtype =None)
label = genfromtxt("../data/ADIENCE/age_label.txt", delimiter = ",", dtype =int)

image_train = all_dir[:13000]
image_test = all_dir[13000:]

label_train = label[:13000]
label_test = label[13000:]

train = np.column_stack((image_train, label_train.T))
test = np.column_stack((image_test, label_test.T))

np.savetxt("../data/ADIENCE/age_train.txt", train, fmt="%s", delimiter = " ")
np.savetxt("../data/ADIENCE/age_test.txt", test, fmt="%s",  delimiter = " ")
