import os
import pdb
import h5py
import pickle
import numpy as np
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import tensorflow as tf
import torchfile
import torch

def read_torch_pt(path):
    ckpt = torch.load(path)
    pdb.set_trace()

if __name__ == "__main__":
    read_torch_pt('/home/nod/Downloads/tb0875_10M.pt')