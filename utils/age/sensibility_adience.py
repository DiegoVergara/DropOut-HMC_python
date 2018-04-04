import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# example data
#y = np.array([76.51,77.78333333,77.57333333,77.38,77.28333333,77.11,77.11,76.8,77.17666667,76.85,76.44333333,76.93666667,76.66333333,76.56,76.68666667,76.99333333,76.58,76.49333333,76.17333333])
#x = np.array([192,381,569,759,946,1136,1324,1513,1702,1890,2079,2267,2455,2644,2832,3022,3211,3399,3588])
#xe = np.array([114,109,104,99,93,88,83,78,72,67,61,57,51,46,40,34,29,22,15])
#ye = np.array([0.335111922,0.136137186,0.175023808,0.252388589,0.928242066,0.522589705,0.426731766,0.075498344,0.52386385,0.186815417,0.456544996,0.352183664,0.31214313,0.285832119,0.160104133,0.162583312,0.360555128,0.221434716,0.305505046])

y = np.array([76.50166667,77.78,77.56166667,77.46666667,77.275,76.96166667,77.06666667,76.965,77.11,76.58333333,76.305,76.82,76.85333333,76.75166667,76.79833333,76.86666667,76.50333333,76.50833333,76.29833333])
x = np.array([192.1183333,347.445,569.6166667,758.5933333,946.8216667,1135.85,1323.493333,1512.168333,1701.738333,1889.621667,2078.958333,2267.32,2425.101667,2643.913333,2833.038333,3022.07,3210.023333,3398.786667,3587.578333])
xe = np.array([114.1866667,108.9883333,103.7933333,98.57333333,93.405,88.15,82.91166667,77.785,72.37333333,67.015,61.58833333,56.53666667,51.10166667,45.66833333,37.38333333,34.31333333,28.42166667,22.06166667,14.67333333])
ye = np.array([0.416433268,0.148593405,0.476966106,0.245248174,0.605995049,0.438516438,0.375108873,0.209069366,0.357435309,0.336610556,0.372437914,0.31962478,0.575314407,0.390559428,0.297819856,0.488166638,0.259126739,0.173599155,0.247904552])
fig, p = plt.subplots()

#y2 = np.array([71.34,75.05,76.24,76.70,77.08,77.83,78.04,77.69,77.75,77.46,77.46,77.43,77.23,77.17,77.17,76.88,76.76,76.44,76.41,75.98,76.27,75.84,75.60,75.75,75.63,75.28,75.23,75.14,74.94,74.71,74.42])
#x2 = np.array([107,290,407,586,690,1097,1415,1655,1867,2064,2178,2324,2455,2527,2607,2658,2768,2802,2854,2887,2941,2982,2995,3047,3109,3179,3210,3230,3297,3427,3498])
y2 = np.array([71.34,75.05,76.24,76.70,77.08,77.83,78.04,77.69,77.75,77.46,77.43,77.23,77.17,77.17,76.44,75.75,75.23,74.94,74.42])
x2 = np.array([107,290,407,586,690,1097,1415,1655,1867,2064,2324,2455,2527,2607,2802,3047,3210,3297,3498])
hfont = {'fontname':'Arial'}
hfont2 = {'fontname':'Arial', 'weight':'bold'}
p.errorbar(x, y, xerr=xe, yerr=ye, label="DHMC")
p.errorbar(x2, y2, xerr=0.0, yerr=0.0, label="LASSO")
p.legend()
plt.xlabel('Number of Non Zero Features',**hfont)
plt.ylabel('Accuracy',**hfont)
plt.title('Sensitivity Analysis - Adience Dataset',**hfont2)
pp = PdfPages('sensitivity_adience.pdf')
plt.savefig(pp, format='pdf')
pp.close()
plt.show()