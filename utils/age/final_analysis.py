import numpy as np
import seaborn as sns#; sns.set(color_codes=True)
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

data = np.array([[47.8,50.9,50.9,51.6,52.3,51.0,50.5,49.2,48.7],
				[48.3,49.2,50.5,52.2,52.3,50.2,50.5,50.0,48.4],
				[47.0,49.4,49.6,50.6,50.8,51.1,48.3,49.3,49.1],
				[46.6,50.3,52.3,51.2,50.7,49.8,50.7,50.1,48.2],
				[43.4,50.3,52.0,51.7,50.3,50.9,49.6,50.5,47.7]])

ax = sns.tsplot(data=data, time=np.linspace(0.1, 0.9, data.shape[1]))

plt.title("Sensitivity Analysis: ADIENCE Data Set", fontsize=10)
plt.xlabel("DropOut Rate", fontsize=10)
plt.ylabel("Accuracy", fontsize=10)
plt.savefig("MNIST_Total.pdf", format='pdf')
#plt.savefig("MNIST_Total.png")
#plt.show()
plt.close()

print(data.mean(axis=0))
print(data.std(axis=0))


#sgld | sghmc
data2 = np.array([[45.6, 47.6],
				[44.5, 45.1],
				[42.6, 45.9],
				[44.4, 43.8],
				[42.9, 45.8]])


print(data2.mean(axis=0))
print(data2.std(axis=0))

