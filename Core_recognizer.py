import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from Recognizer.backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    print(feat)


    # Save the feature embeddings as a numpy array in the current working directory
    current_dir = os.getcwd()
    np.save(os.path.join(current_dir, 'test02.npy'), feat) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='/home/suger01/Downloads/shit.pth')
    parser.add_argument('--img', type=str, default='cropped_test02.jpg')
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)

