import numpy as np
import data as data_module
from utils.inference_utils import SplitComb
from importlib import import_module

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
    
    # Get one sample directly
    data, coord, nzhw = dataset[0]
    print(f"Data shape: {data.shape}")
    print(f"Coord shape: {coord.shape}")
    print(f"nzhw: {nzhw}")

if __name__ == '__main__':
    main()
