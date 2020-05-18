import pickle
import numpy as np
import pdb

def get_data(data_path):
    data = open(data_path,"rb")
    data_dict = pickle.load(data)

    # rgb image
    rgb = data_dict['rgb']

    # depth data, unit is in millimeter
    depth = data_dict['depth']

    # visualzied depth image
    depth_image = data_dict['depth_image']

    pdb.set_trace()

if __name__ == "__main__":
    data_path = '000004.pkl'
    get_data(data_path)