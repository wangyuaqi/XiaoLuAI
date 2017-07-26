import configparser
from datetime import datetime
from random import randrange

import cv2
import itchat
import os
import requests
from bs4 import BeautifulSoup
from itchat.content import *

config = configparser.ConfigParser()
config.read('/home/lucasx/PycharmProjects/XiaoLuAI/res/config.ini')

CHATFILE_DIR = '/home/lucasx/wechat/'
STICKER_DIR = '/home/lucasx/stickers/'
TEMP_IMAGE_DOWNLOAD_DIR = '/tmp/tmpimg/'
GIRL_DIR = '/home/lucasx/girl/'


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING, PICTURE])
def text_reply(msg):
    NickName = itchat.search_friends(userName=msg['FromUserName'])['NickName']
    mkdirs_if_not_exists(CHATFILE_DIR)
    with open(CHATFILE_DIR + str(NickName) + '.txt', mode='a', encoding='utf-8') as f:
        f.write(str(datetime.now()) + ' : ' + msg['Content'] + '\r')
        f.flush()
        f.close()
    # itchat.send('%s: %s' % (msg['Type'], msg['Text']), msg['FromUserName'])
    if msg['Type'] == 'Text':
        itchat.send('[Auto Reply]\r\n' + chat_with_ai(config['Turing']['apikey'], msg['Text']),
                    msg['FromUserName'])
    elif msg['Type'] == 'PICTURE':
        mkdirs_if_not_exists(CHATFILE_DIR + msg['FromUserName'])
        download_wechat_sticker(msg['Content'], CHATFILE_DIR + str(NickName) + '/', str(datetime.now()) + '.gif')
    else:
        itchat.send('[Auto Reply]\r\nSorry, 小璐机器人目前还没有表情识别功能哟，别着急，爸爸正在给我加上啦~嘻嘻', msg['FromUserName'])


@itchat.msg_register([PICTURE, RECORDING, ATTACHMENT, VIDEO])
def download_files(msg):
    msg['Text'](msg['FileName'])
    return '@%s@%s' % ({'Picture': 'img', 'Video': 'vid'}.get(msg['Type'], 'fil'), msg['FileName'])


"""
@itchat.msg_register(['Picture', 'Recording', 'Attachment', 'Video'])
def download_files(msg):
    msg['Text'](msg['FileName'])
    itchat.send('@%s@%s' % ('img' if msg['Type'] == 'Picture' else 'fil', msg['FileName']), msg['FromUserName'])
    return '%s received' % msg['Type']
"""


@itchat.msg_register(FRIENDS)
def add_friend(msg):
    itchat.add_friend(**msg['Text'])  # 该操作会自动将新好友的消息录入，不需要重载通讯录
    itchat.send_msg('Nice to meet you!', msg['RecommendInfo']['UserName'])


@itchat.msg_register([TEXT, PICTURE], isGroupChat=True)
def text_reply(msg):
    mkdirs_if_not_exists(CHATFILE_DIR)
    groupName = msg['User']['NickName']
    # if msg['isAt']:
    with open(CHATFILE_DIR + str(groupName) + '.txt', mode='a', encoding='utf-8') as f:
        if msg['Type'] == 'Text':
            f.write(msg['ActualNickName'] + ' : ' + str(datetime.now()) + '  :  ' + msg['Content'] + '\r')
            f.flush()
            f.close()
            # itchat.send(u'@%s\u2005I received: %s' % (msg['ActualNickName'], msg['Content']), msg['FromUserName'])
        elif msg['Type'] == 'PICTURE':
            download_wechat_sticker(msg['Content'], CHATFILE_DIR + groupName, msg['ActualNickName'] + '.jpg')

    if '大叔' in groupName:
        # itchat.send('[Auto Reply]\r\n@' + msg['ActualNickName'] + '\t' + chat_with_ai('68312595a53e4feb8165cd1335d7be7f', msg['Content']), msg['FromUserName'])
        all_stickers = os.listdir(STICKER_DIR)
        print(STICKER_DIR + str(all_stickers[randrange(len(all_stickers))]))
        itchat.send('@img@' + STICKER_DIR + str(all_stickers[randrange(len(all_stickers))]),
                    toUserName=msg['FromUserName'])


def download_wechat_sticker(message_xml, dirname, filename):
    """
    unfinished
    :param message_xml:
    :param dirname:
    :param filename:
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/'
                      '58.0.3029.110 Chrome/58.0.3029.110 Safari/537.36',
        'Host': 'emoji.qpic.cn',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    msg_xml_soup = BeautifulSoup(message_xml, 'html5lib')
    cdurl = msg_xml_soup.msg.emoji['cdnurl']
    response = requests.get(cdurl, headers=headers)
    mkdirs_if_not_exists(CHATFILE_DIR + dirname)
    if response.status_code == 200:
        with open(CHATFILE_DIR + dirname + filename, mode='wb') as f:
            f.write(response.content)
            f.flush()
            f.close()
            print('Write sticker successfully~')
    else:
        print(response.status_code)


def get_group(search_filter_username):
    """search group by any content filter"""
    return itchat.search_chatrooms(name=search_filter_username)


def weather_service(city):
    """Weather API"""
    weather_url = 'http://api.jirengu.com/weather.php?city=' + city
    reply_content = ''
    response = requests.get(weather_url)
    response.encoding = 'UTF-8'
    if response.status_code == 200 and response.json()['status'] == 'success':
        result = response.json()['results'][0]
        reply_content = result['currentCity'] + ' 的今日PM2.5指数为' + str(result['pm25']) + '\r\n'
        index = result['index']
        for each_item in index:
            temp = each_item['tipt'] + ' : ' + each_item['des'] + '\r\n'
            reply_content += temp
    elif response.status_code == 403:
        print('No access!')
    else:
        print(response.status_code)

    return reply_content


def chat_with_ai(apikey, chat_content):
    """AI chat bot with TuLing robot"""
    req_url = 'http://www.tuling123.com/openapi/api'
    params = {
        "key": apikey,
        "info": chat_content
    }

    reply = '[玫瑰][玫瑰][玫瑰]'
    response = requests.post(req_url, params=params)
    if response.json()['code'] == 100000:  # response type is TEXT
        reply = response.json()['text']
        print(reply)
    elif response.json()['code'] == 200000:  # response type is URL
        reply = response.json()['text'] + '\r\n' + response.json()['url']
    elif response.json()['code'] == 302000:  # response type is NEWS
        news = response.json()['list'][randrange(len(response.json()['list']))]
        reply = response.json()['text'] + '\r\n' + news['article'] + '\r\n' + news['source'] + '\r\n' + news[
            'detailurl']
    elif response.json()['code'] == 308000:  # response type is RECIPE
        recipe = response.json()['list'][randrange(len(response.json()['list']))]
        reply = response.json()['text'] + '\r\n' + recipe['info'] + '\r\n' + recipe['detailurl']
    else:
        print('ERROR' + str(response.json()))

    return reply


def crawl_sticker():
    """crawl stickers from Internet"""
    mkdirs_if_not_exists(STICKER_DIR)
    url_list = ['http://qq.yh31.com/zjbq/0551964.html', 'http://qq.yh31.com/zjbq/0551964_2.html',
                'http://qq.yh31.com/zjbq/0551964_3.html']
    headers = {
        'Host': 'qq.yh31.com',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/'
                      '56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }

    for url in url_list:
        response = requests.get(url, headers=headers)
        if response.status_code not in [404, 403]:
            html_raw = response.text
            soup = BeautifulSoup(html_raw, 'html5lib')
            for each_img in soup.find_all('img'):
                if each_img['src'].startswith('/tp/zjbq/'):
                    img_prefix = 'http://qq.yh31.com'

                    resp = requests.get(img_prefix + each_img['src'])
                    if resp.status_code not in [404, 403]:
                        with open(STICKER_DIR + each_img['src'].split('/')[-1], mode='wb') as f:
                            f.write(resp.content)
                            f.flush()
                            f.close()
                            print('Download Sticker' + each_img['src'].split('/')[-1] + ' Successfully!')
                    else:
                        print('Download Sticker Error!')
        else:
            print('Error!!')


def mkdirs_if_not_exists(directory_):
    """create a new directory if the arg dir not exists"""
    if not os.path.exists(directory_) or not os.path.isdir(directory_):
        os.makedirs(directory_)


def crawl_girl():
    """crawl girl image from Internet"""
    if not os.path.exists(GIRL_DIR) or not os.path.isdir(GIRL_DIR):
        os.makedirs(GIRL_DIR)

    url_list = ['http://www.mm131.com/qingchun/']

    for url in url_list:
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code not in [404, 403]:
            html_raw = response.text
            soup = BeautifulSoup(html_raw, 'html5lib')
            for dd in soup.find_all('dd'):
                img_url = dd.a['href']
                if img_url.startswith('http://www.mm131.com/qingchun/'):
                    resp = requests.get(img_url)
                    if resp.status_code not in [404, 403]:
                        img_node = BeautifulSoup(resp.text, 'html5lib').find_all('div', class_='content-pic')[0].a.img
                        img_resp = requests.get(img_node['src'])
                        if img_resp.status_code not in [404, 403]:
                            img_filename = img_node['src'].split('/')[-2] + img_node['src'].split('/')[-1]
                            with open(GIRL_DIR + img_filename, mode='wb') as f:
                                f.write(img_resp.content)
                                f.flush()
                                f.close()
                                print('Download ' + img_filename + ' successfully!')

        else:
            print('Error!!!')


"""
def group_send(msg_content):
    friendList = itchat.get_friends(update=True)[1:]
    for friend in friendList:
        # itchat.send(msg_content % (friend['DisplayName'] or friend['NickName']), friend['UserName'])
        print(msg_content % (friend['DisplayName'] or friend['NickName']), friend['UserName'])
        time.sleep(.5)
"""


def face_detect(image_path):
    """detect faces with an uploaded local file"""
    req_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    try:
        fr = open(image_path, mode='rb')
        img_face = fr.read()
        params = {
            'api_key': config['FacePP']['api_key'],
            'api_secret': config['FacePP']['api_secret'],
            'return_landmark': 1,
            'image_file': img_face,
            'return_attributes': 'gender,age'
        }
        files = {'image_file': open(image_path, 'rb')}
        headers = {
            'Content-Type': 'multipart/form-data'
        }
        response = requests.post(req_url, headers=headers, params=params, files=files)
        fr.close()
        face = response.json()
        print(face)

        return face
    except Exception as e:
        print('Invalid image')
        print(str(e))


def face_detect_with_url(image_url):
    req_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    try:
        params = {
            'api_key': config['FacePP']['api_key'],
            'api_secret': config['FacePP']['api_secret'],
            'return_landmark': 1,
            'image_url': image_url,
            'return_attributes': 'gender,age'
        }
        headers = {
            'Content-Type': 'multipart/form-data'
        }
        response = requests.post(req_url, headers=headers, params=params, timeout=5)
        face_json = response.json()
        mkdirs_if_not_exists(TEMP_IMAGE_DOWNLOAD_DIR)
        image_save_path = TEMP_IMAGE_DOWNLOAD_DIR + image_url.split('/')[-1]
        fw = open(image_save_path, mode='wb')
        fw.write(requests.get(image_url).content)
        fw.flush()
        fw.close()
        print(str(image_url.split('/')[-1]) + ' has been downloaded successfully~')
        image = cv2.imread(image_save_path)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        i = 0
        for face in face_json['faces']:
            # draw the face outline
            cv2.rectangle(image, (face['face_rectangle']['left'], face['face_rectangle']['top']),
                          (face['face_rectangle']['left'] + face['face_rectangle']['width'],
                           face['face_rectangle']['top'] + face['face_rectangle']['height']), (0, 255, 0), 3)
            # cv2.putText(image, '颜值8.8', face['face_rectangle']['width'], cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, '9.6', (
                face['face_rectangle']['left'] + face['face_rectangle']['width'], face['face_rectangle']['top']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, False)

            # plot contour point
            cv2.circle(image, (face['landmark']['contour_left2']['x'], face['landmark']['contour_left2']['y']), 5,
                       (225, 225, 0), -1)
            cv2.circle(image, (face['landmark']['contour_right2']['x'], face['landmark']['contour_right2']['y']), 5,
                       (225, 225, 0), -1)

            cv2.circle(image, (face['landmark']['contour_chin']['x'], face['landmark']['contour_chin']['y']), 5,
                       (225, 0, 225), -1)

            cv2.circle(image, (face['landmark']['contour_left1']['x'], face['landmark']['contour_left1']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right1']['x'], face['landmark']['contour_right1']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left2']['x'], face['landmark']['contour_left2']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right2']['x'], face['landmark']['contour_right2']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left3']['x'], face['landmark']['contour_left3']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right3']['x'], face['landmark']['contour_right3']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left5']['x'], face['landmark']['contour_left5']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right5']['x'], face['landmark']['contour_right5']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left6']['x'], face['landmark']['contour_left6']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right6']['x'], face['landmark']['contour_right6']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left7']['x'], face['landmark']['contour_left7']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right7']['x'], face['landmark']['contour_right7']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left8']['x'], face['landmark']['contour_left8']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right8']['x'], face['landmark']['contour_right8']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_left9']['x'], face['landmark']['contour_left9']['y']), 5,
                       (225, 0, 225), -1)
            cv2.circle(image, (face['landmark']['contour_right9']['x'], face['landmark']['contour_right9']['y']), 5,
                       (225, 0, 225), -1)

            # plot eye point
            cv2.circle(image, (face['landmark']['left_eye_center']['x'], face['landmark']['left_eye_center']['y']), 5,
                       (225, 0, 0), -1)
            cv2.circle(image, (face['landmark']['right_eye_center']['x'], face['landmark']['right_eye_center']['y']), 5,
                       (225, 0, 0), -1)

            # plot nose point
            cv2.circle(image,
                       (face['landmark']['nose_contour_left3']['x'], face['landmark']['nose_contour_left3']['y']), 5,
                       (0, 0, 225), -1)
            cv2.circle(image,
                       (face['landmark']['nose_contour_right3']['x'], face['landmark']['nose_contour_right3']['y']), 5,
                       (0, 0, 225), -1)

            cv2.circle(image,
                       (face['landmark']['nose_contour_left3']['x'], face['landmark']['nose_contour_left3']['y']), 5,
                       (0, 0, 225), -1)
            cv2.circle(image,
                       (face['landmark']['nose_contour_right3']['x'], face['landmark']['nose_contour_right3']['y']), 5,
                       (0, 0, 225), -1)

            # plot mouth point
            cv2.circle(image, (face['landmark']['mouth_lower_lip_right_contour2']['x'],
                               face['landmark']['mouth_lower_lip_right_contour2']['y']), 5, (0, 0, 225), -1)
            cv2.circle(image, (face['landmark']['mouth_lower_lip_left_contour2']['x'],
                               face['landmark']['mouth_lower_lip_left_contour2']['y']), 5, (0, 0, 225), -1)

            cv2.imshow('image' + str(i), image)
            cv2.waitKey(0)

        return face
    except Exception as e:
        print('Invalid image')
        print(str(e))


if __name__ == '__main__':
    # itchat.auto_login(True)
    # itchat.run()
    # group_send('Hello~')
    crawl_sticker()
    # crawl_girl()
