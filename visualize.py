import matplotlib.pyplot as plt
import matplotlib
import numpy as np

filename = "ggf6202.csv"

x_edges, y_edges = np.linspace(-2.5, 2.5, 50), np.linspace(-3.2, 3.16, 63)
matrix = np.genfromtxt(filename, delimiter=",", skip_header=0)
x = matrix[:,1]
y = matrix[:,2]
my_cmap = matplotlib.cm.get_cmap('jet')
my_cmap.set_under('white')
bin_vals = (x_edges, y_edges)
result, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges), range=None, normed=False, weights=matrix[:, 0])
plt.pcolormesh(result, cmap=my_cmap, vmin=0.1)
plt.xlabel('eta')
plt.ylabel('phi')
cbar = plt.colorbar()
cbar.ax.set_ylabel('pT')
plt.show()
