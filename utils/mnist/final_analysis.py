import numpy as np
import seaborn as sns#; sns.set(color_codes=True)
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

data = np.array([[88.3,90.6,91.1,91.6,91.8,92.1,92.0,91.9,91.7],
				[88.2,90.4,90.9,91.5,91.9,91.9,91.5,92.0,91.7],
				[88.4,90.1,90.9,91.6,91.6,92.1,91.9,91.7,91.8],
				[88.2,90.1,90.9,91.5,91.7,92.0,91.7,92.1,91.5],
				[88.2,90.3,91.1,91.6,91.6,91.5,91.9,91.9,91.6]])

ax = sns.tsplot(data=data, time=np.linspace(0.1, 0.9, data.shape[1]))

plt.title("Sensitivity Analysis: MNIST Data Set", fontsize=10)
plt.xlabel("DropOut Rate", fontsize=10)
plt.ylabel("Accuracy", fontsize=10)
plt.savefig("MNIST_Total.pdf", format='pdf')
#plt.savefig("MNIST_Total.png")
#plt.show()
plt.close()

print(data.mean(axis=0))
print(data.std(axis=0))



data2 = np.array([[90.8, 87.9],
				[90.8, 87.9],
				[91.5, 88.0],
				[90.7, 88.4],
				[90.9, 88.1]])


print(data2.mean(axis=0))
print(data2.std(axis=0))

