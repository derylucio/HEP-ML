from extract_data_from_trees import extract_background_pt
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import simps
from sklearn import metrics


NUM_CRITICAL_POINTS = 100;
ggf_bg_pts, vbf_bg_pts = extract_background_pt()
print ggf_bg_pts.shape
print vbf_bg_pts.shape
critical_points =  np.linspace(np.min(ggf_bg_pts), np.max(ggf_bg_pts), NUM_CRITICAL_POINTS)
efficiencies = np.zeros((NUM_CRITICAL_POINTS, 2))
num_ggf = 1.0*len(ggf_bg_pts)
num_vbf = 1.0*len(vbf_bg_pts)
for i in range(NUM_CRITICAL_POINTS):
	cp = critical_points[i]
	ggf_passes = ggf_bg_pts < cp
	ggf = np.sum(ggf_passes)/num_ggf
	vbf_passes = vbf_bg_pts < cp
	vbf = np.sum(vbf_passes)/num_vbf
	efficiencies[i, 0] = ggf
	efficiencies[i, 1] = vbf
plt.ylabel('True Positive VBF (correctly labeled VBF)')
plt.xlabel('False Positive GGF (incorrectly labeled GGF)')
area = metrics.auc(efficiencies[:,0], efficiencies[:,1])#np.trapz(efficiencies[:,1], x=efficiencies[:,0])
plt.plot(efficiencies[:,0], efficiencies[:,1])
title = "HTSoft Graph:Unequal Classes"
plt.figtext(.4, .5, "AUC : " + str(area))
pp = PdfPages(title + ".pdf")
plt.savefig(pp, format="pdf")
pp.close()
plt.show()