import argparse
import os
import time
import numpy as np
import data as data_module
from importlib import import_module
import shutil
from utils.log_utils import *
import sys
from utils.inference_utils import SplitComb, postprocess, plot_box 
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='ca detection')
parser.add_argument('--model', '-m', metavar='MODEL', default='model.network',
                    help='model')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input', default='', type=str, metavar='data',
                    help='directory to save images (default: none)')
parser.add_argument('--output', default='', type=str, metavar='SAVE',
                    help='directory to save prediction results(default: none)')
parser.add_argument('--test', default=1, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')


def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    test_name = (args.input).split("/")[-1]
    data_dir = (args.input).split("/")[-2]
    save_dir = (args.output).split("/")[-2]

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        net.load_state_dict(checkpoint['state_dict'])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    
    # Use CPU instead of CUDA
    net = net.cpu()
    loss = loss.cpu()
    
    margin = config["margin"]
    sidelen = config["split_size"]

    split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
    import data as data_module
    dataset = data_module.TestDetector(
        data_dir,
        test_name,
        config,
        split_comber=split_comber)
    test_loader = DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=False)

    net.eval()
    namelist = []
    split_comblist = []
    outputlist = []
    featurelist = []
    
    for i, (data, coord, nzhw) in enumerate(test_loader):
        s = time.time()
        nzhw = nzhw[0]
        name = dataset.filenames[i].split('-')[0].split('/')[-1]
        
        # Data is already split into patches, process each patch
        data = data[0]  # Remove batch dimension
        coord = coord[0]  # Remove batch dimension
        
        with torch.no_grad():
            data = Variable(data.cpu())
            coord = Variable(coord.cpu())
            outputlist.append(net(data, coord))
        split_comblist.append(split_comber)
        namelist.append(name)
        featurelist.append([None, nzhw])
        e = time.time()
        
    for i in range(len(outputlist)):
        output = outputlist[i]
        split_comber = split_comblist[i]
        name = namelist[i]
        feature = featurelist[i]
        lbb, nzhw = feature
        if lbb is None:
            lbb = np.array([[0, 0, 0, 0, 0]])
        output = split_comber.combine(output,nzhw=nzhw)
        output = output[0]
        thresh = -3
        pbb, mask = get_pbb(output,thresh,ismask=True)
        
        if len(pbb) > 0:
            pbb = postprocess(pbb, lbb, config, n_test=args.n_test)
            pbb = pbb[pbb[:,0] > 0.05]
            pbb = nms(pbb, 0.05)
        
        print(name, pbb.shape)
        np.save(os.path.join(args.output, name + '_pbb.npy'), pbb)
        np.save(os.path.join(args.output, name + '_lbb.npy'), lbb)


def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


if __name__ == '__main__':
    main()
