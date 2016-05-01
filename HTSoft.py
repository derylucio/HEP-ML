from extract_data_from_trees import extract_background_pt
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages


NUM_CRITICAL_POINTS = 100;
ggf_bg_pts, vbf_bg_pts = extract_background_pt()
critical_points =  np.linspace(np.min(ggf_bg_pts), np.max(ggf_bg_pts), NUM_CRITICAL_POINTS)
efficiencies = np.zeros((NUM_CRITICAL_POINTS, 2))
num_ggf = 1.0*len(ggf_bg_pts)
num_vbf = 1.0*len(vbf_bg_pts)
for i in range(NUM_CRITICAL_POINTS):
	cp = critical_points[i]
	ggf_passes = ggf_bg_pts > cp
	true_pos = np.sum(ggf_passes)/num_ggf
	vbf_passes = vbf_bg_pts > cp
	false_pos = np.sum(vbf_passes)/num_vbf
	efficiencies[i, 0] = true_pos
	efficiencies[i, 1] = false_pos
plt.ylabel('False Positive Rate (incorrectly classified VBF)')
plt.xlabel('True Positive Rate (correctly classified GGF)')
plt.plot(efficiencies[:,0], efficiencies[:,1])
title = "HTSoft Graph"
pp = PdfPages(title + ".pdf")
plt.savefig(pp, format="pdf")
pp.close()
plt.show()