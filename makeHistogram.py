import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.backends.backend_pdf import PdfPages

NUM_BUCKETS = 30
PT_INDEX = 0

tracks = glob.glob("tracks2/*.csv")

def makeHistogram(filename):
    matrix = np.genfromtxt(filename, delimiter=",", skip_header=1)
    pt_values = matrix[:, PT_INDEX]
    title = filename.split("/")[1].split(".")[0]
    plt.hist(pt_values, bins=NUM_BUCKETS)
    plt.title(title)
    pp = PdfPages("Histograms/" + title + '.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.close()


for filename in tracks:
    makeHistogram(filename)


