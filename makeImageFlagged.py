from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import glob

TYPE_INDEX = 4
ETA_INDEX = 1

my_cmap = matplotlib.cm.get_cmap('jet')
my_cmap.set_under('white')
def makeImage(filename):
    matrix = np.genfromtxt(filename, delimiter=",", skip_header=1)
    vbf = matrix[matrix[:, TYPE_INDEX] == 1]
    max_eta = np.max(vbf[:, ETA_INDEX])
    min_eta = np.min(vbf[:, ETA_INDEX])
    matrix[:, ETA_INDEX] = 2.0*( matrix[:, ETA_INDEX] - min_eta)/(max_eta - min_eta) - 1.0   #tranform eta to -1 for min_eta and 1 for max_eta
    background = matrix[matrix[:, TYPE_INDEX] == 0]
    vbf = matrix[matrix[:, TYPE_INDEX] == 1]
    bjet = matrix[matrix[:, TYPE_INDEX] == 2]
    bbellipse = matrix[matrix[:, TYPE_INDEX] == 3]

    #Generate images
    x_positive, x_negative =  2.0*(2.5 - min_eta)/(max_eta - min_eta) - 1.0, 2.0*(-2.5 - min_eta)/(max_eta - min_eta) - 1.0
    x_edges, y_edges = np.linspace(x_negative, x_positive, 50), np.linspace(-3.2, 3.16, 63)
    x_ticks = np.linspace(x_negative, x_positive, 6)
    plt.subplot(2, 2, 1)
    generate_plot(background, x_edges, y_edges, x_ticks, "Background")
    plt.subplot(2, 2, 2)
    generate_plot(bjet, x_edges, y_edges,x_ticks, "Bjet")
    plt.subplot(2, 2, 3)
    generate_plot(bbellipse, x_edges, y_edges, x_ticks, "BBellipse")
    plt.subplot(2, 2, 4)
    generate_plot(vbf, x_edges, y_edges, x_ticks, "VBFJet")
    plt.tight_layout()
    title = filename.split("/")[1].split(".")[0]
    pp = PdfPages("subevent_images/" + title + ".pdf")
    plt.savefig(pp, format="pdf")
    plt.close()
    pp.close()


def generate_plot(mat, x_edges, y_edges, x_ticks, title):
    result, x_end, y_end = np.histogram2d(mat[:, 1], mat[:, 2], bins=(x_edges, y_edges), range=None, normed=False, weights=mat[:, 0])
    plt.xticks(np.linspace(0, 49, 6), [str(int(x*10)/10.0) for x in x_ticks])
    plt.yticks(np.linspace(0, 62, 6), ('-3.15', '-1.89', '-0.63', '0.63', '1.89', '3.15'))
    plt.pcolormesh(np.flipud(result.T), cmap=my_cmap, vmin=0.1)
    plt.xlabel('Rapidity[eta]')
    plt.ylabel('Azimuthal Angle[phi]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('pT')
    plt.title(title)

for filename in glob.glob("tracks/*.csv"):
    makeImage(filename)