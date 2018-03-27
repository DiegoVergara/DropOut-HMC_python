import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
data = pd.read_csv("SGLD_mnist_analysis.csv", header = 0)

data[' GT'] = data[' GT'].astype(int)
columns_dict = {'# mean_0':'0', ' mean_1':'1', ' mean_2':'2', ' mean_3':'3', ' mean_4':'4', ' mean_5':'5', ' mean_6':'6', ' mean_7':'7', ' mean_8':'8', ' mean_9':'9', ' GT': ' GT'}
new_data = data.rename(index=str, columns=columns_dict)
data_list = []

for x in xrange(0,10):
	data_list.append(new_data.loc[new_data[' GT'] == x])


#columns = ['mean_0', ' mean_1', ' mean_2', ' mean_3', ' mean_4', ' mean_5', ' mean_6', ' mean_7', ' mean_8', ' mean_9']

new_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


for x in xrange(0,10):
	temp_mean = data_list[x][new_columns].mean(axis=0)
	temp_var = data_list[x][new_columns].var(axis=0)
	temp_mean.plot(kind = "bar", yerr=temp_var, fontsize = 10)
	plt.title("Histogram: Probability in the MNIST data set for class "+str(x), fontsize=10)
	plt.xlabel("Class", fontsize=10)
	plt.ylabel("Mean of Probability", fontsize=10)
	#plt.savefig("MNIST_mean_"+str(x)+"_histogram.png")
	pp = PdfPages('MNIST_mean_"+str(x)+"_histogram.pdf')
	plt.savefig(pp, format='pdf')
	#plt.show()
	plt.close()
