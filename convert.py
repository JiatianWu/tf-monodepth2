from conversion.convert import SaveModel
import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    dataset_name = 'tello'

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