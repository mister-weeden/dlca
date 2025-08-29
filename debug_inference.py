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


def main():
    model = import_module('model.network')
    config, net, loss, get_pbb = model.get_model()
    
    test_name = "brain_CTA"
    data_dir = "test_image"
    
    margin = config["margin"]
    sidelen = config["split_size"]

    split_comber = SplitComb(sidelen,config['max_stride'],config['stride'],margin,config['pad_value'])
    dataset = data_module.TestDetector(
        data_dir,
        test_name,
        config,
        split_comber=split_comber)
    
    test_loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=False)

    for i, batch in enumerate(test_loader):
        print(f"Batch {i}:")
        print(f"Number of items in batch: {len(batch)}")
        for j, item in enumerate(batch):
            if torch.is_tensor(item):
                print(f"  Item {j}: tensor shape {item.shape}")
            else:
                print(f"  Item {j}: {type(item)} - {item}")
        break

if __name__ == '__main__':
    main()
