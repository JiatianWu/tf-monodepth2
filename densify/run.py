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
import pdb
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

input_frames = "sample_data/frames/"
input_colmap = "sample_data/reconstruction/"
output_folder = "output/"

dump_debug_images = True

# Algorithm parameters. See the paper for details.

tau_high = 0.1
tau_low = 0.1
tau_flow = 0.2
k_I = 5
k_T = 7
k_F = 31
lambda_d = 1
lambda_t = 0.01
lambda_s = 1

num_solver_iterations = 500

class Reconstruction:
    def __init__(self):
        self.cameras = {}
        self.views = {}
        self.points3d = {}
        self.min_view_id = -1
        self.max_view_id = -1
        self.image_folder = ""
    
    def ViewIds(self):
        return list(self.views.keys())
    
    def GetNeighboringKeyframes(self, view_id):
        previous_keyframe = -1
        next_keyframe = -1
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                previous_keyframe = idx
                break
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                next_keyframe = idx
                break
        if previous_keyframe < 0 or next_keyframe < 0:
            return np.array([])
        return [previous_keyframe, next_keyframe]
    
    def GetReferenceFrames(self, view_id):
        kf = self.GetNeighboringKeyframes(view_id)
        if (len(kf) < 2):
            return []
        dist = np.linalg.norm(self.views[kf[1]].Position() -\
                              self.views[kf[0]].Position()) / 2
        pos = self.views[view_id].Position()
        ref = []
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos -\
                              self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos -\
                              self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        return ref

    def GetImage(self, view_id):
        return self.views[view_id].GetImage(self.image_folder)
    
    def GetSparseDepthMap(self, frame_id):
        camera = self.cameras[self.views[frame_id].camera_id]
        view = self.views[frame_id]
        view_pos = view.Position()
        depth_map = np.zeros((camera.height, camera.width), dtype=np.float32)
        for point_id, coord in view.points2d.items():
            pos3d = self.points3d[point_id].position3d
            depth = np.linalg.norm(pos3d - view_pos)
            depth_map[int(coord[1]), int(coord[0])] = depth
        return depth_map
    
    def Print(self):
        print("Found " + str(len(self.views)) + " cameras.")
        for id in self.cameras:
            self.cameras[id].Print()
        print("Found " + str(len(self.views)) + " frames.")
        for id in self.views:
            self.views[id].Print()

class Point:
    def __init__(self):
        self.id = -1
        self.position3d = np.zeros(3, float)
    
            
class Camera:

    def __init__(self):
        self.id = -1
        self.width = 0
        self.height = 0
        self.focal = np.zeros(2,float)
        self.principal = np.zeros(2,float)
        self.model = ""
    
    def Print(self):
        print("Camera " + str(self.id))
        print("-Image size: (" + str(self.width) + \
            ", " + str(self.height) + ")")
        print("-Focal: " + str(self.focal))
        print("-Model: " + self.model)
        print("")

class View:    
    def __init__(self):
        self.id = -1
        self.orientation = Quaternion()
        self.translation = np.zeros(3, float)
        self.points2d = {}
        self.camera_id = -1
        self.name = ""
    
    def IsKeyframe(self):
        return len(self.points2d) > 0
    
    def Rotation(self):
        return self.orientation.rotation_matrix
    
    def Position(self):
        return self.orientation.rotate(self.translation)
    
    def GetImage(self, image_folder):
        mat = cv2.imread(image_folder + "/" + self.name)
        # Check that we loaded correctly.
        assert mat is not None, \
            "Image " + self.name + " was not found in " \
            + image_folder
        return mat
    
    def Print(self):
        print("Frame " + str(self.id) + ": " + self.name)
        print("Rotation: \n" + \
            str(self.Rotation()))
        print("Position: \n" + \
            str(self.Position()))
        print("")
        
def ReadColmapCamera(filename):
    file = open(filename, "r")
    line = file.readline()
    cameras = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            cameras[id_value] = Camera()
            cameras[id_value].id = id_value
            cameras[id_value].model = tokens[1]
            # Currently we're assuming that the camera model
            # is in the SIMPLE_RADIAL format
            assert(cameras[id_value].model == "PINHOLE")
            cameras[id_value].width = int(tokens[2])
            cameras[id_value].height = int(tokens[3])
            cameras[id_value].focal[0] = float(tokens[4])
            cameras[id_value].focal[1] = float(tokens[5])
            cameras[id_value].principal[0] = float(tokens[6])
            cameras[id_value].principal[1] = float(tokens[7])
        line = file.readline()
    return cameras;

def ReadColmapImages(filename):
    file = open(filename, "r")
    line = file.readline()
    views = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            views[id_value] = View()
            views[id_value].id = id_value
            views[id_value].orientation = Quaternion(float(tokens[1]), \
                                                     float(tokens[2]), \
                                                     float(tokens[3]), \
                                                     float(tokens[4]))
            views[id_value].translation[0] = float(tokens[5])
            views[id_value].translation[1] = float(tokens[6])
            views[id_value].translation[2] = float(tokens[7])
            views[id_value].camera_id = int(tokens[8])
            views[id_value].name = tokens[9]
            line = file.readline()
            tokens = line.split()
            views[id_value].points2d = {}
            for idx in range(0, len(tokens) // 3):
                point_id = int(tokens[idx * 3 + 2])
                coord = np.array([float(tokens[idx * 3 + 0]), \
                         float(tokens[idx * 3 + 1])])
                views[id_value].points2d[point_id] = coord
            
            # Read the observations...
        line = file.readline()
    return views
           
def ReadColmapPoints(filename):
    file = open(filename, "r")
    line = file.readline()
    points = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            points[id_value] = Point()
            points[id_value].id = id_value
            points[id_value].position3d = np.array([float(tokens[1]), \
                                        float(tokens[2]), \
                                        float(tokens[3])])
            
        line = file.readline()
    return points
        
            
    
def ReadColmap(poses_folder, images_folder):
    # Read the cameras (intrinsics)
    recon = Reconstruction()
    recon.image_folder = images_folder
    recon.cameras = ReadColmapCamera(poses_folder + "/cameras.txt")
    recon.views = ReadColmapImages(poses_folder + "/images.txt")
    recon.points3d = ReadColmapPoints(poses_folder + "/points3D.txt")
    recon.min_view_id = min(list(recon.views.keys()))
    recon.max_view_id = max(list(recon.views.keys()))
    print("Number of points: " + str(len(recon.points3d)))
    print("Number of frames: " + str(len(recon.views)))
    #assert len(recon.views) == (recon.max_view_id - recon.min_view_id) + 1, "Min\max: " + str(recon.max_view_id) + " " + str(recon.min_view_id)
    return recon

import flow_color

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
        plt.imsave(output_folder + "/image_gradient.png", \
                img_grad_magnitude)
    flow_gradient_magnitude = np.zeros(img_grad_magnitude.shape)
    
    max_reliability = np.zeros(flow_gradient_magnitude.shape)
    i = 0
    for flow in flows:
        magnitude, reliability = GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y)
        if (dump_debug_images):
            plt.imsave(output_folder + "/flow_" + str(i) + ".png", \
                    flow_color.computeImg(flow))            
            plt.imsave(output_folder + "/reliability_" + str(i) + ".png", \
                    reliability)
        flow_gradient_magnitude[reliability > max_reliability] = magnitude[reliability > max_reliability]
        i += 1
        
    if (dump_debug_images):
        plt.imsave(output_folder + "/flow_gradient.png", \
                flow_gradient_magnitude)
    flow_gradient_magnitude = \
        cv2.GaussianBlur(flow_gradient_magnitude,(k_F, k_F),0)
    flow_gradient_magnitude *= img_grad_magnitude
    flow_gradient_magnitude /= flow_gradient_magnitude.max()
    return flow_gradient_magnitude
    
def Canny(soft_edges, image):
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
                    if (m > tau_high and soft_edges[x,y] > tau_flow):
                        seeds.append((x,y))
                        edges[x,y] = 255
                    elif (m > tau_low):
                        edges[x,y] = 1
            else:
                tg67x = tg22x + (ax << 16)
                if (ay > tg67x):
                    if (m > mag[x+1,y] and m >= mag[x-1,y]):
                        if (m > tau_high and soft_edges[x,y] > tau_flow):
                            seeds.append((x,y))
                            edges[x,y] = 255
                        elif (m > tau_low):
                            edges[x,y] = 1
                else:
                    if (int(gx[x,y]) ^ int(gy[x,y]) < 0):
                        if (m > mag[x-1,y+1] and m >= mag[x+1,y-1]):
                            if (m > tau_high and soft_edges[x,y] > tau_flow):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
                    else:
                        if (m > mag[x-1,y-1] and m > mag[x+1,y+1]):
                            if (m > tau_high and soft_edges[x,y] > tau_flow):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
    w = img_gradient.shape[0] - 1
    h = img_gradient.shape[1] - 1
    if (dump_debug_images):
        plt.imsave(output_folder + "/edge_seeds.png", \
            edges == 255)
        plt.imsave(output_folder + "/edge_all_possible.png", \
            edges == 1)
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
    
def GetInitialization(sparse_points, last_depth_map):
    initialization = sparse_points.copy()
    if last_depth_map.size > 0:
        initialization[last_depth_map > 0] = 1.0 / last_depth_map[last_depth_map > 0]
    
    w = edges.shape[0]
    h = edges.shape[1]
    last_known = -1
    first_known = -1
    for col in range(0,w):
        for row in range(0,h):
            if (sparse_points[col, row] > 0):
                last_known = 1.0 / sparse_points[col, row]
            elif (initialization[col, row] > 0):
                last_known = initialization[col, row]
            if (first_known < 0):
                first_known = last_known
            initialization[col, row] = last_known
    initialization[initialization < 0] = first_known
    
    return initialization
    
    
def DensifyFrame(sparse_points, hard_edges, soft_edges, last_depth_map):
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
    initialization = GetInitialization(sparse_points, last_depth_map)
                             
    if (dump_debug_images):
        plt.imsave(output_folder + "/solver_initialization" + ".png", \
                initialization)
        plt.imsave(output_folder + "/sparse_points_" + ".png", \
                sparse_points)
        plt.imsave(output_folder + "/soft_edges_" + ".png", \
                soft_edges)
        plt.imsave(output_folder + "/hard_edges_" + ".png", \
                hard_edges)
    
    for row in range(1,h - 1):
        for col in range(1,w - 1):
            x0[col + row * w] = initialization[col, row]
            # Add the data constraints
            if (sparse_points[col, row] > 0.00):
                A[num_entries, col + row * w] = lambda_d
                b[num_entries] = (1.0 / sparse_points[col, row]) * lambda_d
                num_entries += 1
            elif (last_depth_map.size > 0 and last_depth_map[col, row] > 0):
                A[num_entries, col + row * w] = lambda_t
                b[num_entries] = (1.0 / last_depth_map[col, row]) * lambda_t
                num_entries += 1
    
            # Add the smoothness constraints
            smoothness_weight = lambda_s * min(smoothness[col, row], \
                                               smoothness[col - 1, row])
            if (hard_edges[col, row] == hard_edges[col - 1, row]):
                smoothness_x[col,row] = smoothness_weight
                A[num_entries, (col - 1) + row * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
            
            smoothness_weight = lambda_s * min(smoothness[col,row], \
                                               smoothness[col, row - 1])
            if (hard_edges[col,row] == hard_edges[col, row - 1]):
                smoothness_y[col,row] = smoothness_weight
                A[num_entries, col + (row - 1) * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
    
    
    # Solve the system
    if (dump_debug_images):
        plt.imsave(output_folder + "/solver_smoothness_x_" + ".png", \
                smoothness_x)
        plt.imsave(output_folder + "/solver_smoothness_y_" + ".png", \
                smoothness_y)

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
            dis = x[col + row * w]
            if dis > 0:
                depth[col,row] = 1.0 / dis

    return depth

def TemporalMedian(depth_maps):
    lists = {}
    depth_map = depth_maps[0].copy()
    h = depth_map.shape[0]
    w = depth_map.shape[1]
    for row in range(0,h):
        for col in range(0,w):
            values = []
            for img in depth_maps:
                if (img[row,col] > 0):
                    values.append(img[row, col])
            if len(values) > 0:
                depth_map[row,col] = np.median(np.array(values))
            else:
                depth_map[row,col] = 0
    return depth_map

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
    last_depths = []
    last_depth = np.array([])

    data_path = '/home/nod/project/dso/build/sample/000068.pkl'
    data = open(data_path,"rb")
    data_dict = pickle.load(data)

    rgb = data_dict['rgb']
    depth_pred = data_dict['depth_pred'] * 0.7131490173152352
    depth_dso = np.array(Image.open('/home/nod/project/dso/build/sample/00023.png'), np.float) * 0.00666
    depth_gt = np.array(data_dict['depth_gt'], np.float) / 1000

    rgb_before = np.array(Image.open('/home/nod/project/dso/build/sample/00065.jpg'))
    rgb_after = np.array(Image.open('/home/nod/project/dso/build/sample/00071.jpg'))
    depth_pred_image = np.array(Image.open('/home/nod/project/dso/build/sample/depth_pred.png'))

    base_img = rgb
    flows = []
    flows.append(GetFlow(base_img, rgb_before))
    flows.append(GetFlow(base_img, rgb_after))
    soft_edges = GetSoftEdges(base_img, flows)
    edges = Canny(soft_edges, base_img)

    depth_final = DensifyFrame(depth_dso, edges, soft_edges, last_depth)
    data_dict = {'depth_densify': depth_final,
                 'rgb': rgb,
                 'depth_pred': depth_pred,
                 'depth_dso': depth_dso,
                 'depth_gt': depth_gt,
                 'soft_edges': soft_edges,
                 'edge': edges,
                 'flows': flows}
    dict_file_name = '/home/nod/project/dso/build/sample/00068_densify.pkl'
    f = open(dict_file_name,"wb")
    pickle.dump(data_dict, f)
    f.close()

    # depth_final = DensifyFrame(sparse_points, soft_edges, dense_points) * 100
    # depth_final = depth_gt
    # depth_final = bilateral_filter(depth_pred_image, depth_dso)

    plt.imshow(depth_final)
    plt.show(block=True)
    import pdb; pdb.set_trace()
    pc_path = '/home/nod/project/dso/build/sample/00068.ply'
    generate_pc_kinect(rgb, depth_final, pc_path)