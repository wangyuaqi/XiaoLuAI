from collections import OrderedDict

config = OrderedDict()
config['label_excel_path'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx'
config['face_image_filename'] = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Processed/SCUT-FBP-{0}.jpg'
config['predictor_path'] = "/media/lucasx/Document/ModelZoo/shape_predictor_68_face_landmarks.dat"
config['vgg_face_model_mat_file'] = '/home/lucasx/ModelZoo/vgg-face.mat'
config['scut_fbp_reg_model'] = './model/scut_fbp_dcnn_bayes_reg.pkl'
config['eccv_fbp_reg_model'] = './model/eccv_fbp_dcnn_bayes_reg.pkl'
config['test_ratio'] = 0.2
config['image_size'] = 128
config['batch_size'] = 64
config['epoch'] = 30
config['num_of_components'] = 800
config['eccv_dataset_attribute'] = './eccv_face_attribute.csv'
config['eccv_dataset_split_csv_file'] = \
    '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/eccv2010_split1.csv'
