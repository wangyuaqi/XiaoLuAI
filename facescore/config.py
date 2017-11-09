from collections import OrderedDict

config = OrderedDict()
config['label_excel_path'] = '/home/xulu/DataSet/Face/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx'
config['face_image_filename'] = '/home/xulu/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-{0}.jpg'
config['predictor_path'] = "/home/lucasx/Documents/PretrainedModels/shape_predictor_68_face_landmarks.dat"
config['vgg_face_model_mat_file'] = '/home/xulu/ModelZoo/vgg-face.mat'
config['test_ratio'] = 0.2
config['image_size'] = 128
config['batch_size'] = 64
