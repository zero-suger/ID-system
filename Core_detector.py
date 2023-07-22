from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from Detector.data import cfg_mnet, cfg_re50
from Detector.layers.functions.prior_box import PriorBox
from Detector.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from Detector.models.retinaface import RetinaFace
from Detector.utils.box_utils import decode, decode_landm
import time
import sys

# The ArgumentParser constructor takes several optional parameters, but in your case, you have provided one mandatory parameter

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./Detector/Weights/Resnet50_Final.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

# The check_keys function you provided compares the keys of a pretrained state dictionary with the keys of a model's state dictionary. 

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

#This function is particularly useful when you have a model that was trained and saved with a previous version of PyTorch that added a common prefix like 'module.' to all parameter names in the state dictionary. The remove_prefix function allows you to adapt the keys to match the expected format in the current version of PyTorch, ensuring compatibility when loading the pretrained weights.

def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}



# When you train a model and save it as a .pt file, you are essentially saving the state of the model, including its architecture and the learned weights or parameters. This allows you to reload the model later without having to train it again from scratch.
def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    
    for i in range(1):
        image_path = "./Real_time_image/captured_image/captured_image.jpg" # have to change in ID verify / and also in real time identification   (./Real_time_image/captured_image/captured_image.jpg)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        #show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4] * 100)
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cx = b[0]
                cy = b[1] + 12
                
                #cv2.putText(img_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

                landms
                # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            #save image
            id_img = 'id_img.jpg'
            real_time_img = "real_time_img.jpg"
            
            if os.path.isfile(id_img):
                real_time_img = 'real_time_img.jpg'
            else:
                real_time_img = 'id_img.jpg'
            cv2.imwrite(real_time_img, img_raw)

    #print(dets)

    integer_dets = [[round(dets[0][i]) if i != 4 else dets[0][i] for i in range(len(dets[0]))]] # syntactic suger code from tensor take intager but don't change 4th element to integer bcz it is confidance score.

    #print(integer_dets)
    
    tensor_p = torch.tensor(integer_dets[0][:4])
    int_elements = [int(element) for element in tensor_p.tolist()]
    left, top, right, bottom = int_elements[:4]
    
    face_region = img_raw[top:bottom, left:right]
    filename = 'cropped_test01.jpg'
    new_filename = 'cropped_test02.jpg'

    if os.path.isfile(filename):
        new_filename = 'cropped_test02.jpg'
    else:
        new_filename = 'cropped_test01.jpg'
        
    cv2.imwrite(new_filename, face_region )
    # # Display the resized face region
    # cv2.imshow("Resized Face Region", resized_face_region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()/home/suger01/Downloads/my_id.JPEG
    
    # [[ 14.085062   18.69524   105.895584  105.1904  0.9887931  40.532692 58.710663   84.40704    56.112167   65.93585    72.52689    48.8947 93.028656   82.0481     90.80467  ]] (random tensor from random image I have to change it to integer bcz integer take less space)
    

        