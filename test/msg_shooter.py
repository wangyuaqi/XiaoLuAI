import requests


def send_msg(msg):
    url = 'http://www17.53kf.com/sendmsg.jsp?_=1515119805091'
    headers = {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip,deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Content-Length': '221',
        'CONTENT-TYPE': 'application/x-www-form-urlencoded',
        'Host': 'www17.53kf.com',
        'Origin': 'http://www17.53kf.com',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/63.0.3239.84 Chrome/63.0.3239.84 Safari/537.36'
    }

    payload = {
        'cmd': 'QST',
        'sid': '76761181617',
        'first_khtempid': '76761181617',
        'did': '10104924',
        'dwid': '72078749',
        'msg': msg,
        'gid': '10340335716008',
        'time': '1515119805092',
        'verify_key': '6477bc21f34b4290085122990952566e',
        'style_id': '106137980',
        'fk_msgid': 'msgid_1515119805091',
        'code': ''
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        print('send successfully~')
    else:
        print(response.status_code)


if __name__ == '__main__':
    while True:
        send_msg('test...')
