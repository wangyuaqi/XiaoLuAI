from collections import OrderedDict

cfg = OrderedDict()

cfg['hmt_model'] = './model/hmtnet.pth'
cfg['gnet_model'] = './model/gnet.pth'
cfg['rnet_model'] = './model/rnet.pth'

cfg['images_dir'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Images'
cfg['gender_base_dir'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Gender'
cfg['race_base_dir'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Race'

cfg['SCUT_FBP5500_csv'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/train_test_files/SCUT-FBP5500.csv'
cfg['cv_split_base_dir'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/train_test_files/' \
                           '5_folders_cross_validations_files'
cfg['4_6_split_dir'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/train_test_files/' \
                       'split_of_60%training and 40%testing'
cfg['batch_size'] = 4
