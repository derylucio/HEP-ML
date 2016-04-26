import numpy as np 
from root_numpy import root2rec

def extract_data():
    #Output :
    #data_samples : a NUM_SAMPLES x DIM_PHI x DIM_ETA matrix of event images 
    #labels : a NUM_SAMPLE x 1 vector of labels corresponding to these images 0 for vbf and 1 for ggf

    #based on bin size of 0.1x0.1
    DIM_ETA = 50
    DIM_PHI = 63
    x_edges, y_edges = np.linspace(-2.5, 2.5, 51), np.linspace(-3.2, 3.16, 64)

    leaves = ["trk_pt", "trk_phi", "trk_eta", "trk_e", "trk_code"]
    unprocessed_data = [root2rec("tracks/tree_ggf.root", "outtree", leaves),root2rec("tracks/tree_vbf.root", "outtree", leaves)]
    num_ggf = len(unprocessed_data[0])
    num_vbf = len(unprocessed_data[1])
    total_samples = num_ggf + num_vbf
    labels = np.zeros((total_samples, 1))
    # ggf are labelled 1 and vbf are labeled 0 
    # a -1 / 1 labelling might be preferable depending on architecture eg SVM
    labels[0 : num_ggf] = 1
    # 3dimensional tensor. The first dimension indexes into a particular image of size DIM_PHIxDIM_ETA
    data_samples = np.zeros((total_samples, DIM_PHI, DIM_ETA))
    for i in range(num_ggf):
        result, dum_x_edges, dum_y_edges = np.histogram2d(unprocessed_data[0]["trk_eta"][i], unprocessed_data[0]["trk_phi"][i], bins=(x_edges, y_edges), range=None, normed=False, weights=unprocessed_data[0]["trk_pt"][i])
        data_samples[i, :, :] =  np.flipud(result.T)  # to make eta increase from left to right and phi from bottom up.
    for i in range(num_vbf):
        result, dum_x_edges, dum_y_edges = np.histogram2d(unprocessed_data[1]["trk_eta"][i], unprocessed_data[1]["trk_phi"][i], bins=(x_edges, y_edges), range=None, normed=False, weights=unprocessed_data[1]["trk_pt"][i])
        data_samples[(i + num_ggf), :, :] =  np.flipud(result.T)  # to make eta increase from left to right and phi from bottom up.

    #shuffle the data
    perm = np.random.permutation(total_samples);
    data_samples = data_samples[perm, :, :]
    labels = labels[perm] 
    return data_samples, labels