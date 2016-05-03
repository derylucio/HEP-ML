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
best_accuracy = -1;
for i in range(NUM_CRITICAL_POINTS):
	cp = critical_points[i]
	ggf_passes = ggf_bg_pts < cp
	ggf = np.sum(ggf_passes)/num_ggf
	vbf_passes = vbf_bg_pts < cp
	vbf = np.sum(vbf_passes)/num_vbf
	accuracy = (num_ggf - np.sum(ggf_passes) + np.sum(vbf_passes))/(num_ggf + num_vbf)
	if accuracy > best_accuracy:
		best_accuracy = accuracy
	efficiencies[i, 0] = ggf
	efficiencies[i, 1] = vbf
print best_accuracy  # produced accuracy of around 0.63
plt.xlabel('True Positive VBF (correctly labeled VBF)')
plt.ylabel('False Positive GGF (incorrectly labeled GGF)')
plt.plot(efficiencies[:,1], efficiencies[:,0])
title = "HTSoft Graph"
pp = PdfPages(title + ".pdf")
plt.savefig(pp, format="pdf")
pp.close()
plt.show()