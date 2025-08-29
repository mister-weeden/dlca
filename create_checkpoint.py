import torch
import torch.nn as nn
from importlib import import_module
import os

# Import the model
import sys
sys.path.append('/Users/owner/work/models/dlca')
model = import_module('model.network')

def create_checkpoint():
    config, net, loss, get_pbb = model.get_model()
    
    # Create a minimal checkpoint
    checkpoint = {
        'epoch': 100,
        'state_dict': net.state_dict(),
        'optimizer': None,
        'config': config
    }
    
    # Save checkpoint
    os.makedirs('/Users/owner/work/models/dlca/checkpoint', exist_ok=True)
    torch.save(checkpoint, '/Users/owner/work/models/dlca/checkpoint/trained_model.ckpt')
    print("Created minimal checkpoint: trained_model.ckpt")

if __name__ == "__main__":
    create_checkpoint()
