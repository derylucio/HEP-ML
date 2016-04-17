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
titles  = ["Background", "BJet", "BBellipse", "VBFJet"]
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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
    packaged_matrix = [background, bjet, bbellipse, vbf]
    for data, name, ax in zip(packaged_matrix, titles, axes.flat):
        result, x_end, y_end = np.histogram2d(data[:, 1], data[:, 2], bins=(x_edges, y_edges), range=None, normed=False, weights=data[:, 0])
        im = ax.imshow(np.flipud(result.T), cmap=my_cmap, vmin=0.1, interpolation="nearest")
        ax.set_xticks(np.linspace(0, 49, 6))
        ax.set_yticks(np.linspace(0, 62, 6))
        ax.set_xticklabels([str(int(x*10)/10.0) for x in x_ticks])
        ax.set_yticklabels(('-3.15', '-1.89', '-0.63', '0.63', '1.89', '3.15'))
        ax.set_xlabel('Rapidity[eta]')
        ax.set_ylabel('Azimuthal Angle[phi]')
        ax.set_title(name)

    fig.subplots_adjust(left=0.05, bottom=None, right=0.9, top=None, wspace=0.1, hspace=3)
    cax = fig.add_axes([0.87, 0.101, 0.02, 0.82])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('pT')
    fig.tight_layout()
    title = filename.split("/")[1].split(".")[0]
    pp = PdfPages("subevent_images/" + title + ".pdf")
    plt.savefig(pp, format="pdf")
    plt.close()
    pp.close()


for filename in glob.glob("tracks/*.csv"):
    makeImage(filename)