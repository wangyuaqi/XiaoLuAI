import random
import pandas as pd

from xpinyin import Pinyin
import itchat, time
from itchat.content import *


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    msg.user.send('%s: %s' % (msg.type, msg.text))


@itchat.msg_register([PICTURE, RECORDING, ATTACHMENT, VIDEO])
def download_files(msg):
    msg.download(msg.fileName)
    typeSymbol = {
        PICTURE: 'img',
        VIDEO: 'vid', }.get(msg.type, 'fil')
    return '@%s@%s' % (typeSymbol, msg.fileName)


@itchat.msg_register(FRIENDS)
def add_friend(msg):
    msg.user.verify()
    msg.user.send('Nice to meet you!')


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING, PICTURE])
def text_reply(msg):
    wechat_msgs = read_msgs()
    wechat_friends = get_all_wechat_friends()
    keywords = read_keywords()

    print(msg)

    if msg['Type'] == 'Text':
        flag = False
        for keyword in keywords:
            if keyword in msg['Text']:
                flag = True
                break
        if flag:
            msg_content = wechat_msgs[random.randint(0, len(wechat_msgs) - 1)]

            friend_info = wechat_friends[msg['User']['NickName']]

            if friend_info['sex'] == 1:
                msg_content += '事业有成，越来越帅！'
            elif friend_info['sex'] == 2:
                msg_content += '越来越漂亮！'

            itchat.send(msg_content, toUserName=msg['FromUserName'])
    elif msg['Type'] == 'PICTURE':
        itchat.send("新年快乐!", toUserName=msg['FromUserName'])
    else:
        # itchat.send('[玫瑰][玫瑰][玫瑰]', msg['FromUserName'])
        pass


def get_all_wechat_friends(wechat_xlsx='./wechat.xlsx'):
    """
    get all WeChat friends and return a Python dict
    :param wechat_xlsx:
    :return:
    """
    df = pd.read_excel(wechat_xlsx)
    wechat_friends = {}

    for i in range(len(df)):
        wechat_friends[df['NickName'][i]] = {
            'sex': df['Sex'][i],
            'city': df['City'][i],
            'contactFlag': df['ContactFlag'][i],
            'remarkName': df['RemarkName'][i],
            'signature': df['Signature'][i]
        }

    return wechat_friends


def login_wechat():
    """
    login your WeChat by scanning QR code
    :return:
    """
    itchat.auto_login(enableCmdQR=False, hotReload=True)
    itchat.run(True)


def read_msgs(msg_file='./spring_festival.txt'):
    """
    read all messages by lines from msg_file
    :param msg_file:
    :return:
    """
    with open(msg_file, mode='rt', encoding='utf-8') as f:
        return f.readlines()


def read_keywords(keywords_txt='./keywords.txt'):
    """
    read all keywords
    :param keywords_txt:
    :return:
    """
    with open(keywords_txt, mode='rt', encoding='utf-8') as f:
        return [_.replace('\n', '') for _ in f.readlines()]


def reply_wechat_msg(msg_text, send_to_username):
    """
    reply a message to your friend
    :param msg_text:
    :param send_to_username:
    :return:
    """
    itchat.send(msg_text, toUserName=send_to_username)
    print('send to {0}: {1}'.format(send_to_username, msg_text))


if __name__ == '__main__':
    login_wechat()
