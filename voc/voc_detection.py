import os
import random

import cv2
from lxml import etree

from voc.config import *


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
