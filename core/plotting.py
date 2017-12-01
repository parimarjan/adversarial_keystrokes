import matplotlib.pyplot as plt
import numpy as np

''' 
X and Y are data points
labels are the labels for the x and y axis
colors is an array of the colors for respective lines.
'''
def my_plot(X, Y, x_label, y_label, colors):
	n = len(Y[0])
	for i in range(n):
		y = Y[:, i]
		plt.plot(X, y, colors[i])
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()

## Sample:

# X = np.array([50, 60, 70])
# Y = np.array([[10, 20, 30], 
# 			 [20, 30, 40],
# 			 [30, 40, 50]])
# my_plot(X, Y, "x-axis", "y-axis", ['r-', 'b-', 'g-'])

'''
results are  OrderedDict([('Classifier Name', 'Manhattan0_median_0'), ('Smart k-means++', [0.75, 0.88, 0.96, 0.98])])

results are  OrderedDict([('Classifier Name', 'Manhattan1_median_5'), ('Smart k-means++', [0.61, 0.73, 0.84, 0.92])])

results are  OrderedDict([('Classifier Name', 'Manhattan2_median_10'), ('Smart k-means++', [0.49, 0.65, 0.84, 0.88])])

results are  OrderedDict([('Classifier Name', 'Manhattan3_median_15'), ('Smart k-means++', [0.49, 0.61, 0.76, 0.82])])

results are  OrderedDict([('Classifier Name', 'Manhattan4_median_20'), ('Smart k-means++', [0.45, 0.55, 0.76, 0.82])])

results are  OrderedDict([('Classifier Name', 'Manhattan5_median_25'), ('Smart k-means++', [0.45, 0.47, 0.71, 0.76])])

results are  OrderedDict([('Classifier Name', 'Manhattan6_median_30'), ('Smart k-means++', [0.45, 0.53, 0.69, 0.75])])

results are  OrderedDict([('Classifier Name', 'Manhattan7_median_35'), ('Smart k-means++', [0.41, 0.45, 0.55, 0.61])])

results are  OrderedDict([('Classifier Name', 'Manhattan8_median_40'), ('Smart k-means++', [0.37, 0.47, 0.57, 0.61])])

results are  OrderedDict([('Classifier Name', 'Manhattan9_median_45'), ('Smart k-means++', [0.31, 0.45, 0.59, 0.61])])

results are  OrderedDict([('Classifier Name', 'Manhattan13_median_65'), ('Smart k-means++', [0.24, 0.27, 0.45, 0.53])])
'''

