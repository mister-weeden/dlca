import numpy as np
import torch
from importlib import import_module

def main():
    # Load model
    model = import_module('model.network')
    config, net, loss, get_pbb = model.get_model()
    
    # Create dummy output
    output = torch.randn(1, 32, 32, 32, 5, 5)
    output_np = output.numpy()
    
    print(f"Output shape: {output_np.shape}")
    
    try:
        result = get_pbb(output_np, thresh=-3, ismask=True)
        print(f"Result type: {type(result)}")
        if isinstance(result, tuple):
            pbb, mask_indices = result
            print(f"PBB shape: {pbb.shape}")
            print(f"Mask indices length: {len(mask_indices)}")
        else:
            print(f"Result shape: {result.shape}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
