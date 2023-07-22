# -*- coding: utf-8 -*-
"""ArcFace_ixd.rec_conv_to_jpg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kDAXa_TtfhM9lRY7DhYz6KXUmkyDA4Uc
"""

import os
import cv2
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

path_imgidx = '/home/suger01/Downloads/faces_umd/train.idx'
path_imgrec = '/home/suger01/Downloads/faces_umd/train.rec'

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

output_folder = '/home/suger01/Desktop/ArcFace_PyTorch/celebA'  # Output folder to save the images
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in tqdm(range(30000)):
    header, s = recordio.unpack(imgrec.read_idx(i + 1))
    img = mx.image.imdecode(s).asnumpy()

    label = str(header.label)
    image_folder = os.path.join(output_folder, label)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_path = os.path.join(image_folder, f'{i}.jpg')

    # Convert BGR to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_path, img_rgb)