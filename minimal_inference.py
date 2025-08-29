import numpy as np
import torch
from importlib import import_module
import os

def main():
    print("=== DLCA Inference Demo ===")
    
    # Load model
    print("1. Loading model...")
    model = import_module('model.network')
    config, net, loss, get_pbb = model.get_model()
    
    # Load checkpoint
    print("2. Loading checkpoint...")
    checkpoint_path = "./checkpoint/trained_model.ckpt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        print("   Checkpoint loaded successfully!")
    else:
        print("   No checkpoint found, using random weights")
    
    # Set to evaluation mode
    net.eval()
    
    # Create dummy input (simulating a brain patch)
    print("3. Creating test input...")
    batch_size = 1
    channels = 1
    depth, height, width = 128, 128, 128
    
    # Create dummy image data
    dummy_image = torch.randn(batch_size, channels, depth, height, width)
    
    # Create dummy coordinate data
    coord_depth, coord_height, coord_width = 32, 32, 32  # Downsampled coordinates
    dummy_coord = torch.randn(batch_size, 3, coord_depth, coord_height, coord_width)
    
    print(f"   Image shape: {dummy_image.shape}")
    print(f"   Coord shape: {dummy_coord.shape}")
    
    # Run inference
    print("4. Running inference...")
    with torch.no_grad():
        try:
            output = net(dummy_image, dummy_coord)
            print(f"   Output shape: {output.shape}")
            print("   Inference successful!")
            
            # Apply threshold and get predictions
            thresh = -3
            output_np = output.numpy()
            # Remove batch dimension and reshape to expected format
            output_np = output_np[0]  # Remove batch dimension: (32, 32, 32, 5, 5)
            
            result = get_pbb(output_np, thresh, ismask=True)
            if isinstance(result, tuple):
                pbb, mask_indices = result
                print(f"   Found {len(pbb)} potential detections")
            else:
                pbb = result
                print(f"   Found {len(pbb)} potential detections")
            
            # Save results
            os.makedirs("./prediction", exist_ok=True)
            np.save("./prediction/demo_output.npy", output.numpy())
            np.save("./prediction/demo_predictions.npy", pbb)
            print("   Results saved to ./prediction/")
            
        except Exception as e:
            print(f"   Error during inference: {e}")
    
    print("=== Demo Complete ===")

if __name__ == '__main__':
    main()
