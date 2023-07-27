
import h5py
import numpy as np

if __name__ == '__main__':

    f = h5py.File('../../../../sumin/result/h5_result/Deepfakes.h5', 'r')
    #print(f.keys())
    print(f.get('000_003'))
    #videos = list(f['Original'].keys())