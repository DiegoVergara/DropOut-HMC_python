import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from tensorflow.examples.tutorials.mnist import input_data

#plt.switch_backend('agg')
'''
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
'''

path = "sgld/"
data = pd.read_csv(path+"SGLD_mnist_analysis.csv", header = 0)

data[' GT'] = data[' GT'].astype(int)
columns_dict = {'# mean_0':'0', ' mean_1':'1', ' mean_2':'2', ' mean_3':'3', ' mean_4':'4', ' mean_5':'5', ' mean_6':'6', ' mean_7':'7', ' GT': ' GT', ' var_0':'v0', ' var_1':'v1', ' var_2':'v2', ' var_3':'v3', ' var_4':'v4', ' var_5':'v5', ' var_6':'v6', ' var_7':'v7'}
#columns_dict = {'# mean_0':'0', ' mean_1':'1', ' mean_2':'2', ' mean_3':'3', ' mean_4':'4', ' mean_5':'5', ' mean_6':'6', ' mean_7':'7', ' mean_8':'8', ' mean_9':'9', ' GT': ' GT', ' std_0':'v0', ' std_1':'v1', ' std_2':'v2', ' std_3':'v3', ' std_4':'v4', ' std_5':'v5', ' std_6':'v6', ' std_7':'v7', ' std_8':'v8', ' std_9':'v9'}
new_data = data.rename(index=str, columns=columns_dict)

row = 94 #csv row

data = new_data.iloc[row-2]
#columns = ['mean_0', ' mean_1', ' mean_2', ' mean_3', ' mean_4', ' mean_5', ' mean_6', ' mean_7', ' mean_8', ' mean_9']

columns = ['0', '1', '2', '3', '4', '5', '6', '7']
columns_var = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']


temp_mean = data[columns]
temp_var = data[columns_var].values
temp_mean.plot(kind = "bar", yerr=temp_var, fontsize = 10)
plt.title("Histogram: Probability in the ADIENCE data set for class "+str(data[' GT']), fontsize=10)
plt.xlabel("Class", fontsize=10)
plt.ylabel("Mean of Probability", fontsize=10)
#plt.savefig("MNIST_mean_"+str(row)+"_histogram.png")
#pp = PdfPages(path+"MNIST_mean_"+str(row)+"_histogram.pdf")
plt.savefig(path+"ADIENCE_mean_"+str(row)+"_histogram.pdf", format='pdf')
#plt.show()
plt.close()

'''
mnist = input_data.read_data_sets("../../data/MNIST/", one_hot=True) 

X_train = mnist.train.images
X_test = mnist.test.images
X_test = X_test - X_train.mean(axis=0);
Y_test = np.argmax(mnist.test.labels,axis=1)

test_image = X_test[row-2:row-1]
test_label = Y_test[row-2]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("SGHMC_MNIST_gt_"+str(row)+".png")
#plt.show()
plt.close()
'''
