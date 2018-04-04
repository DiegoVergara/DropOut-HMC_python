import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


path = "sghmc_d_09/"
data = pd.read_csv(path+"per_class.csv", header = 0, sep = ';')
del data['Unnamed: 8']
del data['Unnamed: 9']
del data['Unnamed: 10']
columns_dict = {'# mean_0':'0', 'mean_1':'1', 'mean_2':'2', 'mean_3':'3', 'mean_4':'4', 'mean_5':'5', 'mean_6':'6', 'mean_7':'7'}
new_data = data.rename(index=str, columns=columns_dict)
#columns = ['mean_0', ' mean_1', ' mean_2', ' mean_3', ' mean_4', ' mean_5', ' mean_6', ' mean_7', ' mean_8', ' mean_9']

new_columns = ['0', '1', '2', '3', '4', '5', '6', '7']


temp_mean = new_data.mean(axis=0)
temp_var = new_data.std(axis=0)
temp_mean.plot(kind = "bar", yerr=temp_var, fontsize = 10)
plt.title("Histogram: Accuracy in the ADIENCE data set per class", fontsize=10)
plt.xlabel("Class", fontsize=10)
plt.ylabel("Mean of Accuracy", fontsize=10)
plt.savefig(path+"ADIENCE_class_histogram.pdf", format='pdf')
#plt.show()
plt.close()