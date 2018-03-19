from collections import OrderedDict

cfg = OrderedDict()

cfg['model'] = './model/mtcnet.pth'
cfg['images'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Images'
cfg['SCUT_FBP5500_txt'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/train_test_files/SCUT-FBP5500.txt'
cfg['cv_split_basedir'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/train_test_files/' \
                          '5_folders_cross_validations_files'
cfg['4_6_split_dir']='/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/train_test_files/' \
                     'split_of_60%training and 40%testing'
