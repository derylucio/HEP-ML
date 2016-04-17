import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.backends.backend_pdf import PdfPages

NUM_BUCKETS = 60
PT_INDEX = 0

tracks = glob.glob("old_tracks/*.csv")

def makeHistogram(filename):
    matrix = np.genfromtxt(filename, delimiter=",", skip_header=1)
    pt_values = matrix[:, PT_INDEX]/1000  #convert to MeV
    title = filename.split("/")[1].split(".")[0]
    plt.hist(pt_values, bins=NUM_BUCKETS)
    plt.title(title)
    plt.xlabel('pT[GeV]')
    plt.ylabel('Number of Tracks')
    plt.xlim((0, 500))
    pp = PdfPages("Histograms/" + title + '.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.close()

for filename in tracks:
    makeHistogram(filename)


