import itchat, time
import os
from itchat.content import *
import pandas as pd


@itchat.msg_register([TEXT, MAP, CARD, NOTE, SHARING])
def text_reply(msg):
    itchat.send('%s: %s' % (msg['Type'], msg['Text']), msg['FromUserName'])


@itchat.msg_register([PICTURE, RECORDING, ATTACHMENT, VIDEO])
def download_files(msg):
    msg['Text'](msg['FileName'])
    return '@%s@%s' % ({'Picture': 'img', 'Video': 'vid'}.get(msg['Type'], 'fil'), msg['FileName'])


@itchat.msg_register(FRIENDS)
def add_friend(msg):
    itchat.add_friend(**msg['Text'])  # 该操作会自动将新好友的消息录入，不需要重载通讯录
    itchat.send_msg('Nice to meet you!', msg['RecommendInfo']['UserName'])


@itchat.msg_register(TEXT, isGroupChat=True)
def text_reply(msg):
    if msg['isAt']:
        itchat.send(u'@%s\u2005I received: %s' % (msg['ActualNickName'], msg['Content']), msg['FromUserName'])


def get_and_output_friends_info():
    """
    get friends information and output them as Excel file
    :return:
    """

    itchat.auto_login(hotReload=True)
    # itchat.run()

    friends_list = []
    friends = itchat.get_friends(update=True)[0:]
    for _ in friends:
        print(_)
        friend = {}
        friend['NickName'] = _['NickName']
        friend['ContactFlag'] = _['ContactFlag']
        friend['Sex'] = _['Sex']
        friend['City'] = _['City']
        friend['Province'] = _['Province']
        friend['Signature'] = _['Signature']
        friend['RemarkName'] = _['RemarkName']
        friends_list.append(friend)

        if not os.path.exists('./avatar'):
            os.makedirs('./avatar')

        try:
            img = itchat.get_head_img(userName=_["UserName"])
            if _['RemarkName'] != '':
                fileImage = open("./avatar/" + _['RemarkName'] + ".jpg", 'wb')
            else:
                fileImage = open("./avatar/" + _['PYQuanPin'] + ".jpg", 'wb')
            fileImage.write(img)
            fileImage.close()
        except:
            pass

    df = pd.DataFrame(friends_list)
    df.to_excel(excel_writer='./wechat.xlsx', sheet_name='WeChatFriends', index=False)


if __name__ == '__main__':
    get_and_output_friends_info()
