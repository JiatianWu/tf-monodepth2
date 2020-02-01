from conversion.convert import SaveModel
import yaml

if __name__ == "__main__":
    dataset_name = 'tum'

    if dataset_name == 'tello':
        config_path = 'config/monodepth2_tello.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_pb(ckpt_dir='saved_model/tello_0121/model-124266',
                    pb_path ='saved_model/tello_0121/model-124266.pb')
        # app.test_video(ckpt_dir='saved_model/tum_0121/model-3342926',
        #                input_dir='/noddata/jiatian/data/tello_undistort/pics_01/',
        #                output_dir='/home/jiatian/dataset/tmp_tello_2')
    elif dataset_name == 'tum':
        config_path = 'config/monodepth2_tum.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f)
        app = SaveModel(config=config)
        app.save_savedModel(ckpt_dir='saved_model_tum/0129/model-85718',
                            savedModel_dir ='tmp/',
                            save_tflite=True)
        # app.save_pb(ckpt_dir='saved_model/tum_0121/model-3342926',
        #             pb_path ='saved_model/tum_0121/model-3342926.pb')
        # app.test_video(ckpt_dir='saved_model/tum_0121/model-3342926',
        #                input_dir='/home/jiatian/dataset/tum/sequence_12',
        #                output_dir='/home/jiatian/dataset/tmp_12')
        # app.test_video(ckpt_dir='saved_model/tum_0121/model-257150',
        #                input_dir='/home/jiatian/dataset/tum/sequence_42',
        #                output_dir='/home/jiatian/dataset/tmp3')
        # app.test_video(ckpt_dir='saved_model/tum_0121/model-600014',
        #                input_dir='/home/jiatian/dataset/tum/sequence_49',
        #                output_dir='/home/jiatian/dataset/tmp4')