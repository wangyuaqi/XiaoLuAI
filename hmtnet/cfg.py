import os
from collections import OrderedDict

cfg = OrderedDict()

cfg['scut_fbp5500_root'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/'
cfg['hmt_model'] = './model/hmtnet.pth'
cfg['gnet_model'] = './model/gnet.pth'
cfg['rnet_model'] = './model/rnet.pth'

cfg['images_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Images')
cfg['gender_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Gender')
cfg['race_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Race')

cfg['SCUT_FBP5500_csv'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/SCUT-FBP5500.csv')
cfg['cv_split_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/5_folders_cross_validations_files')
cfg['4_6_split_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/split_of_60%training and 40%testing')
cfg['batch_size'] = 4
cfg['pretrained_vgg_face'] = '/media/lucasx/Document/ModelZoo/vgg_face_pytorch/VGG_FACE.pth'
