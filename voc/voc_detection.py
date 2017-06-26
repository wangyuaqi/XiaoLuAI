import os
import random

import cv2
from lxml import etree

import tensorflow as tf

BASE_DIR = '/home/lucasx/Documents/Dataset/VOC/VOCtrainval_06-Nov-2007/VOC2007/'
ANNOTATION_XML_DIR = BASE_DIR + 'Annotations/'
JPG_IMAGES_DIR = BASE_DIR + 'JPEGImages/'


def parse_boject_location_from_xml(xml_filepath):
    """
    parse xml file and return its location coordinate list
    :param xml_filepath:
    :return:
    """
    parser = etree.XMLParser()
    tree = etree.parse(xml_filepath, parser)
    objects = tree.xpath('//object')
    result = []
    for _ in objects:
        location = {}
        name = _.xpath('name/text()[1]')[0]
        pose = _.xpath('pose/text()[1]')[0]
        truncated = _.xpath('truncated/text()[1]')[0]
        difficult = _.xpath('difficult/text()[1]')[0]
        xmin = int(_.xpath('bndbox/xmin/text()')[0].strip())
        xmax = int(_.xpath('bndbox/xmax/text()')[0].strip())
        ymin = int(_.xpath('bndbox/ymin/text()')[0].strip())
        ymax = int(_.xpath('bndbox/ymax/text()')[0].strip())

        print(name)

        location['xmin'] = xmin
        location['xmax'] = xmax
        location['ymin'] = ymin
        location['ymax'] = ymax
        result.append(location)

    return result


def draw_bbox():
    for each_img_file in os.listdir(JPG_IMAGES_DIR):
        result = parse_boject_location_from_xml(ANNOTATION_XML_DIR + '%s.xml' % str(each_img_file.split('.')[0]))
        image = cv2.imread(os.path.join(JPG_IMAGES_DIR, each_img_file))
        for _ in result:
            r = random.randrange(255)
            g = random.randrange(255)
            b = random.randrange(255)
            image = cv2.rectangle(image, (_['xmin'], _['ymin']), (_['xmax'], _['ymax']), (r, g, b), 3)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_classify_train_and_test_file(classify_filepath):
    with open(classify_filepath, mode='rt', encoding='UTF-8') as f:
        for _ in f.readlines():
            if int(_.split(' ')[-1].strip()) == -1:
                pass
            elif int(_.split(' ')[-1].strip()) == 1:
                pass


def data_initialize():
    # get train absolute filename list and validation filename list
    with open(BASE_DIR + '/ImageSets/Main/train.txt', mode='rt', encoding='UTF-8') as f:
        train_filenames = [JPG_IMAGES_DIR + '%s.jpg' % _.strip() for _ in f.readlines()]
    with open(BASE_DIR + '/ImageSets/Main/val.txt', mode='rt', encoding='UTF-8') as f:
        validation_filenames = [JPG_IMAGES_DIR + '%s.jpg' % _.strip() for _ in f.readlines()]

    print(validation_filenames)


if __name__ == '__main__':
    classify_file_dir = '/home/lucasx/Documents/Dataset/VOC/VOCtrainval_06-Nov-2007/VOC2007/ImageSets/Main/'
    data_initialize()
