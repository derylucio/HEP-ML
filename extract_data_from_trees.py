import numpy as np 
from root_numpy import root2rec

BACKGROUND_INDEX = 0
DIM_ETA = 52
DIM_PHI = 64

def get_unprocesseddata():
    leaves = ["trk_pt", "trk_phi", "trk_eta", "trk_e", "trk_code"]
    unprocessed_data = [root2rec("tracks/tree_ggf.root", "outtree", leaves),root2rec("tracks/tree_vbf.root", "outtree", leaves)]
    return unprocessed_data

def extract_imagedata(whole_img=False):
    #input: boolean to determine whether to extract whole image or just background
    #Output :
    #data_samples : a NUM_SAMPLES x DIM_PHI x DIM_ETA matrix of event images 
    #labels : a NUM_SAMPLE x 2 vector of labels corresponding to these images ggf at index 0 and vbf at index 1

    #based on bin size of 0.1x0.1
    x_edges, y_edges = np.linspace(-2.5, 2.5, DIM_ETA + 1), np.linspace(-3.2, 3.16, DIM_PHI + 1)

    unprocessed_data = get_unprocesseddata()
    num_ggf = len(unprocessed_data[0])
    num_vbf = len(unprocessed_data[1])
    total_samples = num_ggf + num_vbf
    labels = np.zeros((total_samples, 2))
    # ggf are labelled 1 and vbf are labeled 0 
    # a -1 / 1 labelling might be preferable depending on architecture eg SVM
    labels[0 : num_ggf, 0] = 1
    labels[num_ggf:, 1] = 1
    # 3dimensional tensor. The first dimension indexes into a particular image of size DIM_PHIxDIM_ETA
    data_samples = np.zeros((total_samples, DIM_PHI, DIM_ETA))
    for i in range(num_ggf):
        trk_pt  = unprocessed_data[0]["trk_pt"][i]
        trk_phi = unprocessed_data[0]["trk_phi"][i]
        trk_eta = unprocessed_data[0]["trk_eta"][i]
        if  not whole_img:
            trk_pt, trk_phi, trk_eta = get_background_event(trk_pt, unprocessed_data[0]["trk_code"][i], trk_eta, trk_phi)
        result, dum_x_edges, dum_y_edges = np.histogram2d(trk_eta, trk_phi , bins=(x_edges, y_edges), range=None, normed=False, weights=trk_pt)
        data_samples[i, :, :] =  np.flipud(result.T)  # to make eta increase from left to right and phi from bottom up.
    for i in range(num_vbf):
        trk_pt  = unprocessed_data[1]["trk_pt"][i]
        trk_phi = unprocessed_data[1]["trk_phi"][i]
        trk_eta = unprocessed_data[1]["trk_eta"][i]
        if not whole_img:
            trk_pt, trk_phi, trk_eta = get_background_event(trk_pt, unprocessed_data[1]["trk_code"][i], trk_eta, trk_phi)
        result, dum_x_edges, dum_y_edges = np.histogram2d(trk_eta, trk_phi, bins=(x_edges, y_edges), range=None, normed=False, weights=trk_pt)
        data_samples[(i + num_ggf), :, :] =  np.flipud(result.T)  # to make eta increase from left to right and phi from bottom up.

    #shuffle the data
    perm = np.random.permutation(total_samples);
    data_samples = data_samples[perm, :, :]
    labels = labels[perm, :] 
    return data_samples, labels

def get_background_event(track_pts, trk_codes, trk_eta, trk_phi):
    code = trk_codes == BACKGROUND_INDEX
    track_pts = track_pts[code]
    trk_phi = trk_phi[code]
    trk_eta = trk_eta[code]
    return track_pts, trk_phi, trk_eta

def get_background_pt(track_pts, trk_codes, num_samples):
    all_totalpt = np.zeros((num_samples, 1))
    for i in range(num_samples):
        code = trk_codes[i] == BACKGROUND_INDEX
        backgrnd_pt = track_pts[i][code]
        total_bg_pt = np.sum(backgrnd_pt)
        all_totalpt[i] = total_bg_pt
    return all_totalpt

def extract_background_pt():
    unprocessed_data = get_unprocesseddata()
    num_ggf = len(unprocessed_data[0])
    num_vbf = len(unprocessed_data[1])
    ggf_totalpts = get_background_pt(unprocessed_data[0]["trk_pt"], unprocessed_data[0]["trk_code"], num_ggf)
    vbf_totalpts = get_background_pt(unprocessed_data[1]["trk_pt"], unprocessed_data[1]["trk_code"], num_ggf)
    return ggf_totalpts, vbf_totalpts
