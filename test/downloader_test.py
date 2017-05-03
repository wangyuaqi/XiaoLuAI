import os
import urllib

import requests
import tensorflow as tf

WORK_DIRECTORY = '/tmp/mnist_data/'
# SOURCE_URL = "https://github.com/EclipseXuLu/XiaoLuAI/blob/master/res/face_dataset.zip"
SOURCE_URL = ""


def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=256):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
    return local_filename


def maybe_download(url):
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, url.split('/')[-1])
    if not tf.gfile.Exists(filepath):
        urllib.request.urlretrieve(SOURCE_URL, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', url.split('/')[-1], size, 'bytes.')

    return filepath


if __name__ == '__main__':
    # maybe_download(SOURCE_URL)
    extracted_dir_path = '/tmp/'
    if not os.path.exists(extracted_dir_path):
        import zipfile

        zip_ref = zipfile.ZipFile('/home/lucasx/PycharmProjects/XiaoLuAI/res/face_image.zip', 'r')
        zip_ref.extractall(extracted_dir_path)
        zip_ref.close()
