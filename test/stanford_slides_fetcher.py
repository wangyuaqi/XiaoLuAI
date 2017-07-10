# script to download Standford Slides of TensorFlow for DeepLearning Research
import os
import requests
from bs4 import BeautifulSoup


def get_slides():
    # tf4dlr_url = 'http://web.stanford.edu/class/cs20si/syllabus.html?utm_source=mybridge&utm_medium=web&utm_campaign=read_more'
    cs231n_url = 'http://cs231n.stanford.edu/syllabus.html'
    cs224d_url = 'http://cs224d.stanford.edu/syllabus.html'
    cs224n_url = 'http://web.stanford.edu/class/cs224n/syllabus.html'
    soup = BeautifulSoup(requests.get(cs224n_url, timeout=10).text, 'html5lib')
    for a in soup.find_all('a'):
        if 'slides' in a.get_text():
            with open('./Stanford/CS224n/%s' % a['href'].split('/')[-1], mode='wb') as f:
                base_url = 'http://web.stanford.edu/class/cs224n/'
                f.write(requests.get(base_url + a['href'], timeout=20).content)
                f.flush()
                print('%s has been downloaded successfully~' % a['href'].split('/')[-1])


if __name__ == '__main__':
    get_slides()
