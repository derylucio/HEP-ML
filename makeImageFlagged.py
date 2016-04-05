from scipy import linalg
import numpy as np
import glob

ETA_START = 24
PHI_START = 31
NUM_ROWS_ETA = 50
NUM_COLS_PHI = 63

def makeImage(filename):
    result_0 = np.zeros((NUM_ROWS_ETA, NUM_COLS_PHI))
    result_1 = np.zeros((NUM_ROWS_ETA, NUM_COLS_PHI))
    result_2 = np.zeros((NUM_ROWS_ETA, NUM_COLS_PHI))
    result_3 = np.zeros((NUM_ROWS_ETA, NUM_COLS_PHI))
    result_list = [result_0, result_1, result_2, result_3]
    with open(filename) as fp:
        next(fp)
        for line in fp:
            entry = line.split(",")
            pT_val = float(entry[0])
            #entries range from 2.5 to -2.5 for eta. Say you have -2.36. Multiplying by 10 and converting into a float gives -23.
            #We add the offset of ETA_START (24) to get the right position in the matrix : 1 = 24 - 23.
            row = ETA_START + int(float(entry[1])*10)
            col = PHI_START - int(float(entry[2])*10)
            result_list[int(entry[4])][row][col] += pT_val
    np.savetxt("nonecsv/" + filename.split("/")[1], result_list[0], delimiter=",")
    np.savetxt("vbfcsv/" + filename.split("/")[1], result_list[1], delimiter=",")
    np.savetxt("bjetcsv/" + filename.split("/")[1], result_list[2], delimiter=",")
    np.savetxt("bbellipsecsv/" + filename.split("/")[1], result_list[3], delimiter=",")


for filename in glob.glob("tracks2/*.csv"):
    makeImage(filename)