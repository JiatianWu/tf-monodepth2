from conversion.convert import SaveModel
import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    dataset_name = 'xilinx'

    if dataset_name == 'tello':
        config_path = 'config/monodepth2_tello.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_savedModel(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model_nyu/0210_maxdepth10_2/model-252578',
                            savedModel_dir ='tmp_depth_maxdepth10/',
                            save_tflite=True)
        # app.save_pb(ckpt_dir='saved_model/tello_0121/model-124266',
        #             pb_path ='saved_model/tello_0121/model-124266.pb')
        # app.test_video(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model_nyu/0210_maxdepth10_2/model-252578',
        #                input_dir='/noddata/jiatian/data/tello_undistort/pics_01/',
        #                output_dir='/home/jiatian/dataset/tmp_test/tello_01_nyu_maxdepth2')
    elif dataset_name == 'tum':
        config_path = 'config/monodepth2_tum.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        # app.save_savedModel(ckpt_dir='saved_model_tum/quantize_friendly/model-3257210',
        #                     savedModel_dir ='tmp_depth_2/',
        #                     save_tflite=True)
        # app.save_pb(ckpt_dir='saved_model/tum_0121/model-3342926',
        #             pb_path ='saved_model/tum_0121/model-3342926.pb')
        app.test_video(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model_nyu/0206/model-6288126',
                       input_dir='/home/jiatian/dataset/tum/sequence_42',
                       output_dir='/home/jiatian/dataset/tmp_test/tum_42_nyu')
        # app.test_video(ckpt_dir='saved_model/tum_0121/model-257150',
        #                input_dir='/home/jiatian/dataset/tum/sequence_42',
        #                output_dir='/home/jiatian/dataset/tmp3')
        # app.test_video(ckpt_dir='saved_model/tum_0121/model-600014',
        #                input_dir='/home/jiatian/dataset/tum/sequence_49',
        #                output_dir='/home/jiatian/dataset/tmp4')
    elif dataset_name == 'nyu':
        config_path = 'config/monodepth2_nyu.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        # app.save_savedModel(ckpt_dir='saved_model_tum/quantize_friendly/model-3257210',
        #                     savedModel_dir ='tmp_depth_2/',
        #                     save_tflite=True)
        # app.save_pb(ckpt_dir='saved_model/tum_0121/model-3342926',
        #             pb_path ='saved_model/tum_0121/model-3342926.pb')
        app.test_video(ckpt_dir='saved_model_nyu/0204_finetune/model-3509788',
                       input_dir='/home/jiatian/dataset/tello',
                       output_dir='/home/jiatian/dataset/tmp/tmp_tello_kinect')
    elif dataset_name == 'nod':
        config_path = 'config/noddepth_nyu.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_pb(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model_nyu/0210_maxdepth10_2/model-4798946',
                    pb_path ='saved_model/tflite_test/tmp_nod_test_0217/saved_model.pb')
        # app.save_savedModel(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model_nyu/0210_maxdepth10_2/backup/model-505154',
        #                     savedModel_dir ='saved_model/tflite_test/tmp_nod_test_0217/saved_model.pb',
        #                     save_tflite=True)
    elif dataset_name == 'nod_test':
        config_path = 'config/noddepth_nyu_test.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        # app.save_pb(ckpt_dir='saved_model/ckpt_640_480/model-756002',
        #             pb_path='saved_model/ckpt_320_240/saved_model.pb')
        # app.save_savedModel(ckpt_dir='saved_model/ckpt_640_480/model-756002',
        #                     savedModel_dir ='saved_model/tflite_320_240/',
        #                     save_tflite=True)
        # app.save_savedModel(ckpt_dir='saved_model/ckpt_640_480/model-756002',
        #                     savedModel_dir ='saved_model/tflite_320_240_test/',
        #                     save_tflite=False)
        app.save_savedModel(ckpt_dir='saved_model/ckpt_640_480_bilinear/model-2268002',
                            savedModel_dir ='saved_model/tflite_320_240_bilinear/',
                            save_tflite=True)
        # app.test_video(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model/ckpt_nod/0213_640_480/model-756002',
        #                input_dir='/home/jiatian/dataset/nod_device/Demo_Record2/nodvi/device/data/images2',
        #                output_dir='/home/jiatian/dataset/tmp/tmp_nod_in')
    elif dataset_name == 'kitti':
        config_path = 'config/monodepth2_kitti.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.build_restore_model(ckpt_dir='saved_model/ckpt_tf_monodepth2_640_192_pt/model.latest')

        # drive_list = ['2011_09_26_drive_0091_sync_02', '2011_09_26_drive_0020_sync_02', '2011_09_26_drive_0022_sync_02']
        # root_dir = '/home/nod/datasets/kitti'
        # output_dir = '/home/nod/datasets/kitti_eval_1'
        # for dir in sorted(os.listdir(root_dir)):
        #     if 'drive' in dir and dir == drive_list[2]:
        #         dir_path = root_dir + '/' + dir
        #         output_path = output_dir + '/' + dir 
        #         app.test_dir_depth_pose(input_dir=dir_path,
        #                                 output_dir=output_path)
        dir_path = '/home/nod/project/TrianFlow/data/demo'
        output_path = '/home/nod/project/TrianFlow/data/eval'
        app.test_dir(input_dir=dir_path, output_dir=output_path)

    elif dataset_name == 'lyft':
        config_path = 'config/noddepth_lyft.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.test_video(ckpt_dir='/home/nod/project/tf-monodepth2/saved_model/ckpt_tf_monodepth2_640_192_nopt/model.latest',
                       input_dir='/home/nod/datasets/kitti/2011_09_26_drive_0106_sync_03',
                       output_dir='/home/nod/datasets/kitti/eval_plasma')
    elif dataset_name == 'nod_device':
        config_path = 'config/noddepth_nyu_fullRes.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.build_default_depth_model(ckpt_dir='saved_model/ckpt_640_480/model-756002')

        dir_path = '/home/nod/datasets/nod/RB3/RB3_Demo_Record_15/nodvi/device/data/images0_undistorted'
        output_path = '/home/nod/datasets/nod/RB3/RB3_Demo_Record_15/nodvi/device/data/images0_depth'
        app.test_nod_dir(input_dir=dir_path, output_dir=output_path)
    elif dataset_name == 'xilinx_postprocess':
        config_path = 'config/noddepth_xilinx_vga.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_xilinx_pb_postprocess(ckpt_dir='saved_model/xilinx_640_480/model-756002',
                                       pb_path ='saved_model/xilinx_640_480/saved_model.pb')
    elif dataset_name == 'xilinx_nosigmoid':
        config_path = 'config/noddepth_xilinx_vga.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_xilinx_pb_postprocess_nosigmoid(ckpt_dir='saved_model/xilinx_640_480/model-756002',
                                                 pb_path ='saved_model/xilinx_640_480/saved_model_nosigmoid.pb')
    elif dataset_name == 'xilinx':
        config_path = 'config/noddepth_xilinx_vga.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_xilinx_pb(ckpt_dir='saved_model/xilinx_640_480/model-2268002',
                           pb_path ='saved_model/xilinx_640_480/saved_model_xilinx.pb')
