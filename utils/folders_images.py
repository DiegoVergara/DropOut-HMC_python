import os
import glob
import shutil
import numpy as np
from numpy import genfromtxt
#age_dict = {0 : '0_2', 1 :'4_6', 2 : '8_12', 3 : '15_20', 4: '25_32', 5 : '38_43', 6 : '48_53', 7 : '60_100'}
age_dict = {0 : '0', 1 :'1', 2 : '2', 3 : '3', 4: '4', 5 : '5', 6 : '6', 7 : '7'}
all_dir = genfromtxt("../data/ADIENCE/dataset.txt", delimiter = ",", dtype =None)
label = genfromtxt("../data/ADIENCE/age_label.txt", delimiter = ",", dtype =int)

image_train = all_dir[:13000]
image_test = all_dir[13000:]

label_train = label[:13000]
label_test = label[13000:]

train_dir = "../data/ADIENCE/train"
test_dir = "../data/ADIENCE/test"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

for i, image_dir in enumerate(image_train):
	dst_dir = train_dir+"/"+age_dict[label_train[i]]
	if not os.path.exists(dst_dir):
	    os.makedirs(dst_dir)
	for jpgfile in glob.iglob(image_dir):
	    shutil.copy(jpgfile, dst_dir)

for i, image_dir in enumerate(image_test):
	dst_dir = test_dir+"/"+age_dict[label_test[i]]
	if not os.path.exists(dst_dir):
	    os.makedirs(dst_dir)
	for jpgfile in glob.iglob(image_dir):
	    shutil.copy(jpgfile, dst_dir)

for x in age_dict:
	dst_dir = train_dir+"/"+age_dict[x]
	for index, oldfile in enumerate(glob.glob(dst_dir+"/*.jpg"), start=0):
	    newfile = dst_dir+"/"+age_dict[x]+'.{}.jpg'.format(index)
	    os.rename(oldfile,newfile)

	dst_dir = test_dir+"/"+age_dict[x]
	for index, oldfile in enumerate(glob.glob(dst_dir+"/*.jpg"), start=0):
		newfile = dst_dir+"/"+age_dict[x]+'.{}.jpg'.format(index)
		os.rename(oldfile,newfile)
