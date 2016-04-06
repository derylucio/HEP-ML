import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import numpy as np
import glob

my_cmap = matplotlib.cm.get_cmap('jet')
x_edges, y_edges = np.linspace(-2.5, 2.5, 50), np.linspace(-3.2, 3.16, 63)
bin_vals = (x_edges, y_edges)
my_cmap.set_under('white')
plt.xticks(np.linspace(0, 49, 6), ('-2.5', '-1.5', '-0.5', '0.5', '1.5', '2.5'))
plt.yticks(np.linspace(0, 62, 6), ('-3.15', '-1.89', '-0.63', '0.63', '1.89', '3.15'))
for filename in glob.glob("tracks2/*.csv"):
    matrix = np.genfromtxt(filename, delimiter=",", skip_header=0)
    x = matrix[:,1]
    y = matrix[:,2]
    result, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges), range=None, normed=False, weights=matrix[:, 0])
    plt.pcolormesh( np.flipud(result.T), cmap=my_cmap, vmin=0.1)
    plt.xlabel('eta')
    plt.ylabel('phi')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('pT')
    title = filename.split("/")[1].split(".")[0]
    pp = PdfPages("images/" + title + ".pdf")
    plt.savefig(pp, format="pdf")
    plt.close()
    pp.close()
