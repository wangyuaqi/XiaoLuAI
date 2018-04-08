import os
from collections import OrderedDict

cfg = OrderedDict()

cfg['scut_fbp5500_root'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/'
cfg['hmt_model'] = './model/hmtnet.pth'
cfg['gnet_model'] = './model/gnet.pth'
cfg['rnet_model'] = './model/rnet.pth'

cfg['scutfbp_images_dir'] = os.path.join(os.path.abspath(os.path.dirname(
    cfg['scut_fbp5500_root']) + os.path.sep + "..") + '/SCUT-FBP/Faces')

cfg['scutfbp5500_images_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Images')
cfg['gender_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Gender')
cfg['race_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Race')

cfg['SCUT_FBP5500_csv'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/SCUT-FBP5500.csv')
cfg['cv_split_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/5_folders_cross_validations_files')
cfg['4_6_split_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/split_of_60%training and 40%testing')
cfg['batch_size'] = 32
cfg['pretrained_vgg_face'] = '/media/lucasx/Document/ModelZoo/vgg_m_face_bn_dag.pth'
cfg['dlib_model'] = os.path.join(
    os.path.abspath(cfg['pretrained_vgg_face']) + os.path.sep + '.') + '/shape_predictor_68_face_landmarks.dat'
