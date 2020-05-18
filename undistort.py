# Copyright 2019 Nod Labs
import os
import cv2
import numpy as np
import pdb

class Undistort():
    def __init__(self, input_dir, output_dir):
        # # device 33-d cam 0
        # scale = 0.5
        # fx = 583.780379265
        # fy = 581.614977546
        # cx = 651.918915441
        # cy = 393.602999407
        # d1 = 0.000455566535289
        # d2 = -0.0107723719394
        # d3 = 0.0126745953652
        # d4 = -0.00432266278961
        # self.K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        # self.D = np.array([[d1], [d2], [d3], [d4]])
        # self.compute_params(400, 640)

        # # device 33-d cam 1
        # scale = 0.5
        # fx = 582.67655407
        # fy = 580.348189312
        # cx = 656.558940024
        # cy = 388.944643504
        # d1 = -0.00381739132533
        # d2 = -0.000706352642801
        # d3 = 0.00384560146593
        # d4 = -0.00147758671058
        # self.K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        # self.D = np.array([[d1], [d2], [d3], [d4]])
        # self.compute_params(400, 640)

        # # device 33-d cam 2
        # scale = 0.5
        # fx = 578.5268625564958
        # fy = 577.1383787694714
        # cx = 648.2596757253186
        # cy = 386.23774307470114
        # d1 = 0.0005514600100266717
        # d2 = -0.002507127470220687
        # d3 = 0.0052494247618550365
        # d4 = -0.0019580885945972458
        # self.K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        # self.D = np.array([[d1], [d2], [d3], [d4]])
        # self.compute_params(400, 640)

        # # device 35
        # self.K = np.array([[288.15212989202263, 0.0, 327.18426388880783], [0.0, 287.69154276527473, 195.08868502968176], [0.0, 0.0, 1.0]])
        # self.D = np.array([[-0.00022113678021126206], [-0.0011227150767492507], [0.006647502459197323], [-0.0031347580009617264]])
        # self.compute_params(400, 640)

        # self.K = np.array([[501.00957821, 0.0, 713.40403809], [0.0, 499.15119612, 378.33847811], [0.0, 0.0, 1.0]])
        # self.D = np.array([[0.0], [0.0], [0.0], [0.0]])

        # # media
        # scale = 1.0
        # fx = 578.5268625564958
        # fy = 577.1383787694714
        # cx = 648.2596757253186
        # cy = 386.23774307470114
        # d1 = 0.0005514600100266717
        # d2 = -0.002507127470220687
        # d3 = 0.0052494247618550365
        # d4 = -0.0019580885945972458
        # self.K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        # self.D = np.array([[d1], [d2], [d3], [d4]])
        # self.compute_params(720, 1280)

        # # media
        # scale = 1.0
        # fx = 4.0669569252287715e+02
        # fy = 4.0731320035278435e+02
        # cx = 3.2116956527212687e+02
        # cy = 2.3026206908453682e+02
        # d1 = -4.1207439602712981e-01
        # d2 = 2.5192441892001954e-01
        # d3 = -1.2074621211881384e-03
        # d4 = 3.8698981317186141e-04
        # d5 = -8.9281355012049396e-02
        # self.K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        # self.D = np.array([[d1], [d2], [d3], [d4]])
        # self.compute_params(480, 640)

        # device 35-1 cam 0
        # new_K:
        # fx = 238.52264234
        # fy = 238.15329999
        # cx = 332.14860019
        # cy = 195.73207196
        scale = 1.0
        fx = 286.92994997677204
        fy = 286.4856509324264
        cx = 324.67953012335556
        cy = 197.07953340996804
        d1 = -0.00022113678021126206
        d2 = -0.0011227150767492507
        d3 = 0.006647502459197323
        d4 = -0.0031347580009617264
        self.K = np.array([[fx * scale, 0.0, cx * scale], [0.0, fy * scale, cy * scale], [0.0, 0.0, 1.0]])
        self.D = np.array([[d1], [d2], [d3], [d4]])
        self.height = 400
        self.width = 640
        self.compute_params(self.height, self.width)

        # self.width_resize = int(1280 / 4)
        # self.height_resize = int(800 / 4)
        # self.dim_resize = (self.width_resize, self.height_resize)
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_params(self, h, w):
        DIM = (w, h)

        self.dim1 = (640, 400)
        self.dim2 = self.dim1
        self.dim3 = self.dim1
        self.scaled_K = self.K * self.dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        self.scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.scaled_K, self.D, self.dim2, np.eye(3), balance=0.0)

    def run(self, image):
        img = image
        h, w = img.shape[:2]
        self.compute_params(h, w)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.scaled_K, self.D, np.eye(3), self.new_K, self.dim3, cv2.CV_16SC2)
        dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D,(w,h),1,(w,h))

        # undistort
        # dst = cv2.fisheye.undistortImage(img, self.K, self.D)

        # crop the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]

        # cv2.imshow('orig', img)
        # cv2.imshow('dst', dst)
        # cv2.waitKey(0)

        return dst

    def undistort(self):
        img_list = os.listdir(self.input_dir)
        img_list = sorted(img_list)
        for img in img_list:
            img_path = self.input_dir + img
            img_data = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
            try:
                img_undistorted = self.run(img_data)
            except:
                continue
            img_path_output = self.output_dir + img

            # img_final = cv2.resize(img_undistorted, (1280, 800), interpolation = cv2.INTER_AREA)
            img_final = img_undistorted
            # cv2.imshow('test', img_final)
            # cv2.waitKey(0)
            cv2.imwrite(img_path_output, img_final)
            print(img)

if __name__ == "__main__":
    input_dirlist = ['/noddata/sneha/0429_1_JiatianSets/0429_LH_TM_MM_OfficeSyncSpace_2/home/sneha/ssd/0429_LH_TM_MM_OfficeSyncSpace_2/nodvi/device/data/images0/',
                     '/noddata/sneha/0429_1_JiatianSets/0429_LH_TM_MM_OfficeCommonSpace_3/home/sneha/ssd/0429_LH_TM_MM_OfficeCommonSpace_3/nodvi/device/data/images0/',
                      '/noddata/sneha/0429_1_JiatianSets/0429_LH_TM_MM_OfficeKitchen_4/home/sneha/ssd/0429_LH_TM_MM_OfficeKitchen_4/nodvi/device/data/images0/',
                      '/noddata/sneha/0429_1_JiatianSets/0429_LH_TM_MM_DemoRoom_5/home/sneha/ssd/0429_LH_TM_MM_DemoRoom_5/nodvi/device/data/images0/']
    output_dirlist = ['/home/jiatian/dataset/nod_device/scene_2/images0/',
                      '/home/jiatian/dataset/nod_device/scene_3/images0/',
                      '/home/jiatian/dataset/nod_device/scene_4/images0/',
                      '/home/jiatian/dataset/nod_device/scene_5/images0/']
    input_dir = '/noddata/sneha/0429_1_JiatianSets/0429_LH_TM_MM_OfficeMyDesk_1/nodvi/device/data/images0/'
    output_dir = '/home/jiatian/dataset/nod_device/scene_1/images0/'
    for idx in range(0, 4):
        input_dir = input_dirlist[idx]
        output_dir = output_dirlist[idx]
        instance = Undistort(input_dir, output_dir)
        instance.undistort()