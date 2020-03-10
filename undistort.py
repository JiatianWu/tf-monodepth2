# Copyright 2019 Nod Labs
import os
import cv2
import numpy as np

class Undistort():
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.K = np.array([[286.92994997677204, 0.0, 324.67953012335556], [0.0, 286.4856509324264, 197.07953340996804], [0.0, 0.0, 1.0]])
        self.D = np.array([[-0.00022113678021126206], [-0.0011227150767492507], [0.006647502459197323], [-0.0031347580009617264]])

        # self.K = np.array([[501.00957821, 0.0, 713.40403809], [0.0, 499.15119612, 378.33847811], [0.0, 0.0, 1.0]])
        # self.D = np.array([[0.0], [0.0], [0.0], [0.0]])

        
        # self.width_resize = int(1280 / 4)
        # self.height_resize = int(800 / 4)
        # self.dim_resize = (self.width_resize, self.height_resize)

    def run(self, image):
        img = image
        h, w = img.shape[:2]
        DIM = (w, h)

        dim1 = (640, 400)
        dim2 = dim1
        dim3 = dim1

        scaled_K = self.K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim2, np.eye(3), balance=0)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D,(w,h),1,(w,h))

        # undistort
        # dst = cv2.fisheye.undistortImage(img, self.K, self.D)

        # crop the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]

        cv2.imshow('orig', img)
        cv2.imshow('dst', dst)
        # cv2.waitKey(0)

        return dst

    def undistort(self):
        img_list = os.listdir(self.input_dir)
        img_list = sorted(img_list)
        for img in img_list:
            img_path = self.input_dir + img
            img_data = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
            img_undistorted = self.run(img_data)
            img_path_output = self.output_dir + img

            # img_final = cv2.resize(img_undistorted, (1280, 800), interpolation = cv2.INTER_AREA)
            img_final = img_undistorted
            # cv2.imshow('test', img_final)
            # cv2.waitKey(0)
            cv2.imwrite(img_path_output, img_final)
            print(img)

if __name__ == "__main__":
    input_dir = '/home/jiatian/dataset/nod_device/Demo_Record300_1/nodvi/device/data/images0/'
    output_dir = '/home/jiatian/dataset/nod_device/Demo_Record300_1/nodvi/device/data/images0_undistorted/'
    instance = Undistort(input_dir, output_dir)
    instance.undistort()