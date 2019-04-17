# prepare data for evaluating detection performance
import os
from lxml import etree


def mkdirs_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def prepare_groundtruth_files(anno_xml_dir="/home/xulu/DataSet/VOCdevkit/VOC2007/Annotations",
                              voc_test_txt="/home/xulu/DataSet/VOCdevkit/VOC2007/ImageSets/Layout/test.txt"):
    mkdirs_if_not_exist('./groundtruths')
    with open(voc_test_txt, mode='rt', encoding='utf-8') as f:
        for f_id in f.readlines():
            tree = etree.parse(os.path.join(anno_xml_dir, '{0}.xml'.format(f_id)))
            objects = tree.xpath('//object')

            lines = []
            for object in objects:
                lines.append(' '.join([object.xpath('name/text()')[0], object.xpath('bndbox/xmin/text()')[0],
                                       object.xpath('bndbox/ymin/text()')[0], object.xpath('bndbox/xmax/text()')[0],
                                       object.xpath('bndbox/ymax/text()')[0]]))

            with open('./groundtruths/{0}.txt'.format(f_id), mode='wt', encoding='utf-8') as f:
                f.write('\r'.join(lines))
                f.flush()
                f.close()
            print('write file {0}.txt successfully...'.format(f_id))


def prepare_detection_files(results_dir='/home/xulu/Project/ObjectDetection-OneStageDet/yolo/results'):
    mkdirs_if_not_exist('./detections')
    for txt in os.listdir(results_dir):
        if txt.endswith('.txt'):
            pass




if __name__ == '__main__':
    prepare_groundtruth_files()
