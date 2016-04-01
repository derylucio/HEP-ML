from scipy import linalg
import numpy as np
import glob

ETA_START = 24
PHI_START = 31
NUM_ROWS_ETA = 50
NUM_COLS_PHI = 63

def makeImage(filename):
    result = np.zeros((NUM_ROWS_ETA, NUM_COLS_PHI))
    with open(filename) as fp:
        next(fp)
        for line in fp:
            entry = line.split(",")
            pT_val = float(entry[0])
            #entries range from 2.5 to -2.5 for eta. Say you have -2.36. Multiplying by 10 and converting into a float gives -23.
            #We add the offset of ETA_START (24) to get the right position in the matrix : 1 = 24 - 23.
            row = ETA_START + int(float(entry[1])*10)
            col = PHI_START - int(float(entry[2])*10)
            result[row][col] += pT_val
    new_filename = "imagecsv/" + filename.split("/")[1]
    np.savetxt(new_filename, result, delimiter=",")

for filename in glob.glob("tracks2/*.csv"):
    makeImage(filename)