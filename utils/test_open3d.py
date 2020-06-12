import os
import pdb
import h5py
import pickle
import numpy as np
from scipy.io import loadmat
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import tensorflow as tf

from bilateral_filter import bilateral_filter
from tools import *

def resave_image(path):
    image = Image.open(path)
    image.save(path[:-4] + '.png')

class CameraPose:

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = list(map(int, metastr.split()))
            # import pdb; pdb.set_trace()
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def write_trajectory(traj, filename):
    with open(filename, 'w') as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(' '.join(map(str, x.metadata)) + '\n')
            f.write('\n'.join(
                ' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
            f.write('\n')

def rgbd_odometry_default():
    test_data_folder = '/home/nod/project/Open3D/examples/TestData/'
    pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
        test_data_folder + 'camera_primesense.json')
    print(pinhole_camera_intrinsic.intrinsic_matrix)

    source_color = o3d.io.read_image(test_data_folder + 'RGBD/color/00000.jpg')
    source_depth = o3d.io.read_image(test_data_folder + 'RGBD/depth/00000.png')
    target_color = o3d.io.read_image(test_data_folder + 'RGBD/color/00001.jpg')
    target_depth = o3d.io.read_image(test_data_folder + 'RGBD/depth/00001.png')
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            target_color, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            target_rgbd_image, pinhole_camera_intrinsic)

    option = o3d.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    # [success_color_term, trans_color_term, info] = o3d.odometry.compute_rgbd_odometry(
    #         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
    #         odo_init, o3d.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term, info] = o3d.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
            odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    # if success_color_term:
    #     print("Using RGB-D Odometry")
    #     print(trans_color_term)
    #     import pdb; pdb.set_trace()
    #     source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
    #             source_rgbd_image, pinhole_camera_intrinsic)
    #     source_pcd_color_term.transform(trans_color_term)
    #     o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term])

    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        import pdb; pdb.set_trace()
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
                source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term],
                                        zoom=0.48,
                                        front=[0.0999, -0.1787, -0.9788],
                                        lookat=[0.0345, -0.0937, 1.8033],
                                        up=[-0.0067, -0.9838, 0.1790])

def rgbd_odometry_nyu():
    test_data_folder = '/home/nod/datasets/nyudepthV2/test_kitchen/'
    pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
        test_data_folder + 'camera_primesense.json')
    print(pinhole_camera_intrinsic.intrinsic_matrix)

    idx = 0
    odo_log = []
    cam_to_world = np.eye(4)
    meta_str = str(idx) + ' ' + str(idx) + ' ' + str(idx + 1)
    odo_log.append(CameraPose(meta_str, cam_to_world))
    for idx in range(0, 103):
        source_idx = str(idx).zfill(6)
        source_color = o3d.io.read_image(test_data_folder + 'color/' + source_idx + '.jpg')
        source_depth = o3d.io.read_image(test_data_folder + 'depth/' + source_idx + '.png')
        np.asarray(source_depth)[np.asarray(source_depth) > np.percentile(np.asarray(source_depth), 80)] = 0

        target_idx = str(idx + 1).zfill(6)
        target_color = o3d.io.read_image(test_data_folder + 'color/' + target_idx + '.jpg')
        target_depth = o3d.io.read_image(test_data_folder + 'depth/' + target_idx + '.png')
        np.asarray(target_depth)[np.asarray(target_depth) > np.percentile(np.asarray(target_depth), 80)] = 0
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                source_color, source_depth)
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                target_color, target_depth)
        target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                target_rgbd_image, pinhole_camera_intrinsic)

        option = o3d.odometry.OdometryOption()
        odo_init = np.identity(4)
        print(option)

        [success_hybrid_term, trans_hybrid_term, info] = o3d.odometry.compute_rgbd_odometry(
                source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
                odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

        if success_hybrid_term:
            print("Using Hybrid RGB-D Odometry")
            print(trans_hybrid_term)
            meta_str = str(idx + 1) + ' ' + str(idx + 1) + ' ' + str(idx + 2)
            cam_to_world = np.dot(cam_to_world, trans_hybrid_term)
            odo_log.append(CameraPose(meta_str, cam_to_world))
            # source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            #         source_rgbd_image, pinhole_camera_intrinsic)
            # source_pcd_hybrid_term.transform(trans_hybrid_term)
            # o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
        else:
            print("FAIL ", idx)
            return
    write_trajectory(odo_log, '/home/nod/datasets/nyudepthV2/test_kitchen/odometry.log')

def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation[0, :, :])
        xyzs.append(cam_to_world[:3, 3])
    return xyzs

def tsdf():
    test_data_folder = '/home/nod/datasets/nyudepthV2/test_kitchen/'
    camera_intrinsics = o3d.io.read_pinhole_camera_intrinsic(
        test_data_folder + 'camera_primesense.json')
    camera_poses = read_trajectory(test_data_folder + 'odometry.log')
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    # volume = o3d.integration.UniformTSDFVolume(
    #     length=4.0,
    #     resolution=512,
    #     sdf_trunc=0.04,
    #     color_type=o3d.integration.TSDFVolumeColorType.RGB8,
    # )

    for i in range(0, 103, 1):
    # for i in range(2):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(
            test_data_folder + 'color/{:06d}.jpg'.format(i))
        depth = o3d.io.read_image(
            test_data_folder + 'depth/{:06d}.png'.format(i))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        np.asarray(depth)[np.asarray(depth) > np.percentile(np.asarray(depth), 80)] = 0
        np.asarray(depth)[np.asarray(depth) < np.percentile(np.asarray(depth), 20)] = 0
        volume.integrate(
            rgbd,
            camera_intrinsics,
            camera_poses[i].pose,
        )

    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    o3d.visualization.draw_geometries([voxel_pcd])

    print("Extract voxel-aligned debugging voxel grid")
    voxel_grid = volume.extract_voxel_grid()
    o3d.visualization.draw_geometries([voxel_grid])

    print("Extract point cloud")
    pcd = volume.extract_point_cloud()
    o3d.visualization.draw_geometries([pcd])

def overlay_pc():
    print("Testing camera in open3d ...")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print(intrinsic.intrinsic_matrix)
    print(o3d.camera.PinholeCameraIntrinsic())
    x = o3d.camera.PinholeCameraIntrinsic(640, 480, 518.8579, 519.4696, 325.5824, 253.7362)
    print(x)
    print(x.intrinsic_matrix)
    o3d.io.write_pinhole_camera_intrinsic("test.json", x)
    y = o3d.io.read_pinhole_camera_intrinsic("test.json")
    print(y)
    print(np.asarray(y.intrinsic_matrix))

    print("Read a trajectory and combine all the RGB-D images.")
    pcds = []
    test_data_folder = '/home/nod/datasets/nyudepthV2/test_kitchen/'
    trajectory = o3d.io.read_pinhole_camera_trajectory(
        test_data_folder + 'odometry.log')
    o3d.io.write_pinhole_camera_trajectory("test.json", trajectory)
    print(trajectory)
    print(trajectory.parameters[0].extrinsic)
    print(np.asarray(trajectory.parameters[0].extrinsic))
    for i in range(23, 80, 5):
        color = o3d.io.read_image(
            test_data_folder + 'color/{:06d}.jpg'.format(i))
        depth = o3d.io.read_image(
            test_data_folder + 'depth/{:06d}.png'.format(i))
        np.asarray(depth)[np.asarray(depth) > np.percentile(np.asarray(depth), 50)] = 0
        im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im, trajectory.parameters[i].intrinsic,
            trajectory.parameters[i].extrinsic)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    print("")
    
if __name__ == "__main__":
    # read_trajectory('/home/nod/project/Open3D/examples/TestData/RGBD/odometry.log')
    # rgbd_odometry_nyu()
    # resave_image('/home/nod/datasets/nyudepthV2/test/d-1315403270.612296-3850931981.pgm')
    tsdf()
    # overlay_pc()