from __future__ import division
from __future__ import print_function
import os, scipy.io
import requests
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tifffile

import model

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

input_dir = 'simple_test/input.tiff'
output_dir = 'simple_test/output.png'
checkpoint_dir = './checkpoint/'

if not os.path.isdir('simple_test'):
    os.makedirs('simple_test')

print('Downloading sample image (16-bit tiff)...')
download_file_from_google_drive('14x1Oila4qz3DBN9pxqQ9SnG42UwaL_uB', 'simple_test/input.tiff')

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = model.unet(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded', checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

ratio = 100

in_img = tifffile.imread(input_dir)
input_full = np.expand_dims(np.float32(in_img/65535.0),axis = 0) * ratio

input_full = np.minimum(input_full, 1.0)

output = sess.run(out_image, feed_dict={in_image: input_full})
output = np.minimum(np.maximum(output, 0), 1)

output = output[0, :, :, :]

scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(output_dir)

