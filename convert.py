from conversion.convert import SaveModel
import yaml
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    dataset_name = 'nod_test'

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
        # app.save_pb(ckpt_dir='/home/nod/project/tf-monodepth2/saved_model/ckpt_640_480/model-756002',
        #             pb_path='/home/nod/project/tf-monodepth2/saved_model/tflite_640_480/saved_model.pb')
        app.save_savedModel(ckpt_dir='/home/nod/project/tf-monodepth2/saved_model/ckpt_640_480/model-756002',
                            savedModel_dir ='saved_model/tflite_640_480',
                            save_tflite=True)
        # app.test_video(ckpt_dir='/home/jiatian/project/tf-monodepth2/saved_model/ckpt_nod/0213_640_480/model-756002',
        #                input_dir='/home/jiatian/dataset/nod_device/Demo_Record2/nodvi/device/data/images2',
        #                output_dir='/home/jiatian/dataset/tmp/tmp_nod_in')