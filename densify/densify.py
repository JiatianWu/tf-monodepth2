
import cv2
from cv2 import DISOpticalFlow
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import sys
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm

sys.path.append('../utils')
from bilateral_filter import bilateral_filter

tau_high = 0.1
tau_low = 0.1
tau_flow = 0.2
k_I = 5
k_T = 7
k_F = 31
lambda_sp = 1
lambda_dp = 0.0
lambda_s = 1

num_solver_iterations = 1000

dis = DISOpticalFlow.create(2)
def GetFlow(image1, image2):
    flow = np.zeros((image1.shape[0], image1.shape[1], 2), np.float32)
    flow = dis.calc(\
        cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY),\
        cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY), flow)
    return flow

def AbsoluteMaximum(images):
    assert(len(images) > 0)
    output = images[0]
    for i in range(1,len(images)):
        output[np.abs(images[i]) > np.abs(output)] = images[i][np.abs(images[i]) > np.abs(output)]
    return output

def GetImageGradient(image):
    xr,xg,xb = cv2.split(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))
    yr,yg,yb = cv2.split(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5))
    img_grad_x = AbsoluteMaximum([xr,xg,xb])
    img_grad_y = AbsoluteMaximum([yr,yg,yb])
    
    return img_grad_x, img_grad_y

def GetGradientMagnitude(img_grad_x, img_grad_y):
    img_grad_magnitude = cv2.sqrt((img_grad_x * img_grad_x) \
                                  + (img_grad_y * img_grad_y))
    return img_grad_magnitude

def GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y):
    x1,x2 = cv2.split(cv2.Sobel(flow,cv2.CV_64F,1,0,ksize=5))
    y1,y2 = cv2.split(cv2.Sobel(flow,cv2.CV_64F,0,1,ksize=5))
    flow_grad_x = AbsoluteMaximum([x1,x2])
    flow_grad_y = AbsoluteMaximum([y1,y2])
    flow_gradient_magnitude = cv2.sqrt((flow_grad_x * flow_grad_x) \
                                   + (flow_grad_y * flow_grad_y))
    reliability = np.zeros((flow.shape[0], flow.shape[1]))

    for x in range(0, flow.shape[0]):
        for y in range(1, flow.shape[1]):
            magn = (img_grad_x[x,y] * img_grad_x[x,y]) + \
                (img_grad_y[x,y] * img_grad_y[x,y])
            gradient_dir = np.array((img_grad_y[x,y], img_grad_x[x,y]))
            if (np.linalg.norm(gradient_dir) == 0):
                reliability[x,y] = 0
                continue
            gradient_dir = gradient_dir / np.linalg.norm(gradient_dir)
            center_pixel = np.array((x,y))
            p0 = center_pixel + gradient_dir
            p1 = center_pixel - gradient_dir
            if p0[0] < 0 or p1[0] < 0 or p0[1] < 0 or p1[1] < 0 \
                or p0[0] >= flow.shape[0] or p0[1] >= flow.shape[1] or \
                p1[0] >= flow.shape[0] or p1[1] >= flow.shape[1]:
                reliability[x,y] = -1000
                continue
            f0 = flow[int(p0[0]), int(p0[1])].dot(gradient_dir)
            f1 = flow[int(p1[0]), int(p1[1])].dot(gradient_dir)
            reliability[x,y] = f1 - f0

    return flow_gradient_magnitude, reliability

def GetSoftEdges(image, flows):
    img_grad_x, img_grad_y = GetImageGradient(image)
    img_grad_magnitude = GetGradientMagnitude(img_grad_x, img_grad_y)
    if (dump_debug_images):
        plt.imsave(output_folder + "/image_gradient_" + recon.views[frame].name, \
                img_grad_magnitude)
    flow_gradient_magnitude = np.zeros(img_grad_magnitude.shape)
    
    max_reliability = np.zeros(flow_gradient_magnitude.shape)
    i = 0
    for flow in flows:
        magnitude, reliability = GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y)
        if (dump_debug_images):
            plt.imsave(output_folder + "/flow_" + str(i) + "_" + recon.views[frame].name, \
                    flow_color.computeImg(flow))            
            plt.imsave(output_folder + "/reliability_" + str(i) + "_" + recon.views[frame].name, \
                    reliability)
        flow_gradient_magnitude[reliability > max_reliability] = magnitude[reliability > max_reliability]
        i += 1
        
    if (dump_debug_images):
        plt.imsave(output_folder + "/flow_gradient_" + recon.views[frame].name, \
                flow_gradient_magnitude)
    flow_gradient_magnitude = \
        cv2.GaussianBlur(flow_gradient_magnitude,(k_F, k_F),0)
    flow_gradient_magnitude *= img_grad_magnitude
    flow_gradient_magnitude /= flow_gradient_magnitude.max()
    return flow_gradient_magnitude

def Canny(image):
    image = cv2.GaussianBlur(image, (k_I, k_I), 0)
    xr,xg,xb = cv2.split(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))
    yr,yg,yb = cv2.split(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5))
    img_gradient = cv2.merge((AbsoluteMaximum([xr,xg,xb]),AbsoluteMaximum([yr,yg,yb])))
    
    TG22 = 13573
    
    gx,gy = cv2.split(img_gradient * (2**15))
    mag = cv2.sqrt((gx * gx) \
                    + (gy * gy))
    seeds = []
    edges = np.zeros(mag.shape)
    for x in range(1, img_gradient.shape[0] - 1):
        for y in range(1, img_gradient.shape[1] - 1):
            ax = int(abs(gx[x,y]))
            ay = int(abs(gy[x,y])) << 15
            tg22x = ax * TG22
            m = mag[x,y]
            if (ay < tg22x):
                if (m > mag[x,y-1] and\
                   m >= mag[x,y+1]):
                    #suppressed[x,y] = m
                    if (m > tau_high):
                        seeds.append((x,y))
                        edges[x,y] = 255
                    elif (m > tau_low):
                        edges[x,y] = 1
            else:
                tg67x = tg22x + (ax << 16)
                if (ay > tg67x):
                    if (m > mag[x+1,y] and m >= mag[x-1,y]):
                        if (m > tau_high):
                            seeds.append((x,y))
                            edges[x,y] = 255
                        elif (m > tau_low):
                            edges[x,y] = 1
                else:
                    if (int(gx[x,y]) ^ int(gy[x,y]) < 0):
                        if (m > mag[x-1,y+1] and m >= mag[x+1,y-1]):
                            if (m > tau_high):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
                    else:
                        if (m > mag[x-1,y-1] and m > mag[x+1,y+1]):
                            if (m > tau_high):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1

    w = img_gradient.shape[0] - 1
    h = img_gradient.shape[1] - 1
    while len(seeds) > 0:
        (x,y) = seeds.pop()
        
        if (x < w and y < h and edges[x+1,y+1] == 1):
            edges[x+1,y+1] = 255
            seeds.append((x+1,y+1))
        if (x > 0 and y < h and edges[x-1,y+1] == 1):
            edges[x-1,y+1] = 255
            seeds.append((x-1,y+1))
        if (y < h and edges[x,y+1] == 1):
            edges[x,y+1] = 255
            seeds.append((x,y+1))
        if (x < w and y > 0 and edges[x+1,y-1] == 1):
            edges[x+1,y-1] = 255
            seeds.append((x+1,y-1))
        if (x > 0 and y > 0 and edges[x-1,y-1] == 1):
            edges[x-1,y-1] = 255
            seeds.append((x-1,y-1))
        if (y > 0 and edges[x,y-1] == 1):
            edges[x,y-1] = 255
            seeds.append((x,y-1))
        if (x < w and edges[x+1,y] == 1):
            edges[x+1,y] = 255
            seeds.append((x+1,y))
        if (x > 0 and edges[x-1,y] == 1):
            edges[x-1,y] = 255
            seeds.append((x-1,y))
    edges[edges == 1] = 0
    return edges

def GetInitialization(sparse_points, dense_points):
    initialization = sparse_points.copy()
    
    w = sparse_points.shape[0]
    h = sparse_points.shape[1]
    last_known = -1
    for col in range(0,w):
        for row in range(0,h):
            if (sparse_points[col, row] > 0):
                last_known = 1.0 / sparse_points[col, row]
            elif (dense_points[col, row] > 0):
                last_known = 1.0 / dense_points[col, row]
            initialization[col, row] = last_known
    
    return initialization

def DensifyFrame(sparse_points, soft_edges, dense_points):
    w = sparse_points.shape[0]
    h = sparse_points.shape[1]
    num_pixels = w * h
    A = scipy.sparse.dok_matrix((num_pixels * 3, num_pixels), dtype=np.float32)
    A[A > 0] = 0
    A[A < 0] = 0
    b = np.zeros(num_pixels * 3, dtype=np.float32)
    x0 = np.zeros(num_pixels, dtype=np.float32)
    num_entries = 0
    
    smoothness = np.maximum(1 - soft_edges, 0)
    smoothness_x = np.zeros((w,h), dtype=np.float32)
    smoothness_y = np.zeros((w,h), dtype=np.float32)
    initialization = GetInitialization(sparse_points, dense_points)
    
    for row in range(1,h - 1):
        for col in range(1,w - 1):
            x0[col + row * w] = initialization[col, row]
            # Add the data constraints
            if (sparse_points[col, row] > 0.00):
                A[num_entries, col + row * w] = lambda_sp
                b[num_entries] = (1.0 / sparse_points[col, row]) * lambda_sp
                num_entries += 1
            elif (dense_points[col, row] > 0):
                A[num_entries, col + row * w] = lambda_dp 
                b[num_entries] = (1.0 / dense_points[col, row]) * lambda_dp
                num_entries += 1
    
            # Add the smoothness constraints
            smoothness_weight = lambda_s * min(smoothness[col, row], \
                                               smoothness[col - 1, row])
            if (sparse_points[col, row] > 0.00 and sparse_points[col - 1, row] > 0.0):
                smoothness_x[col,row] = smoothness_weight
                A[num_entries, (col - 1) + row * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
            
            smoothness_weight = lambda_s * min(smoothness[col,row], \
                                               smoothness[col, row - 1])
            if (sparse_points[col, row] > 0.00 and sparse_points[col, row - 1] > 0.0):
                smoothness_y[col,row] = smoothness_weight
                A[num_entries, col + (row - 1) * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1

    [x,info] = scipy.sparse.linalg.cg(A.transpose() * A, \
                                      A.transpose() * b, x0, 1e-05, num_solver_iterations)
    if info < 0:
        print("====> Error! Illegal input!")
    elif info > 0:
        print("====> Ran " + str(info) + " solver iterations.")
    else:
        print("====> Solver converged!")
    
    depth = np.zeros(sparse_points.shape, dtype=np.float32)

    # Copy back the pixels
    for row in range(0,h):
        for col in range(0,w):
            dis = x[col + row*w]
            if dis > 0.0:
                depth[col,row] = 1.0 / dis

    return depth

def generate_pointcloud(rgb, depth, intrinsics=None, ply_file=None):
    points = []
    if intrinsics is not None:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u, :]
            Z = depth[v, u]
            if Z==0:
                continue
            if intrinsics is not None:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
            else:
                X = u
                Y = v
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                property uchar alpha
                end_header
                %s
                '''%(len(points),"".join(points)))
    file.close()

def generate_pc_kinect(rgb, depth, pc_pred_path):
    P_rect = np.eye(3, 3)
    P_rect[0,0] = 400.516317
    P_rect[0,2] = 320.171183
    P_rect[1,1] = 400.410970
    P_rect[1,2] = 243.274495

    # build_output_dir(output_folder)
    generate_pointcloud(rgb, depth, intrinsics=P_rect, ply_file=pc_pred_path)

if __name__ == "__main__":
    data_path = '/home/nod/project/dso/build/sample/000068.pkl'
    data = open(data_path,"rb")
    data_dict = pickle.load(data)

    rgb = data_dict['rgb']
    depth_pred = data_dict['depth_pred'] * 0.7131490173152352
    depth_gt = data_dict['depth_gt']

    depth_pred_image = np.array(Image.open('/home/nod/project/dso/build/sample/depth_pred.png'))
    depth_dso = np.array(Image.open('/home/nod/project/dso/build/sample/00023.png'), np.float) * 0.00666

    sparse_points = depth_dso
    dense_points = depth_pred
    soft_edges = Canny(rgb)

    # depth_final = DensifyFrame(sparse_points, soft_edges, dense_points) * 100
    # depth_final = depth_gt
    depth_final = bilateral_filter(depth_pred_image, depth_dso)

    plt.imshow(depth_final)
    plt.show(block=True)
    import pdb; pdb.set_trace()
    pc_path = '/home/nod/project/dso/build/sample/00068.ply'
    generate_pc_kinect(rgb, depth_final, pc_path)