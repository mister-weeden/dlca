import numpy as np
import nibabel as nib
import os

# Create a smaller test brain CTA image
def create_small_test_image():
    # Create a smaller 3D volume
    shape = (64, 64, 64)
    image = np.random.randint(0, 1000, shape).astype(np.int16)
    
    # Add some brain-like structure
    center = np.array(shape) // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                if dist < 25:  # Brain region
                    image[i,j,k] = np.random.randint(200, 800)
                else:  # Background
                    image[i,j,k] = np.random.randint(0, 100)
    
    # Create NIfTI image
    affine = np.eye(4)
    nii_img = nib.Nifti1Image(image, affine)
    
    # Save the test image
    os.makedirs('/Users/owner/work/models/dlca/test_image', exist_ok=True)
    nib.save(nii_img, '/Users/owner/work/models/dlca/test_image/brain_CTA_small.nii.gz')
    print("Created small test image: brain_CTA_small.nii.gz")

if __name__ == "__main__":
    create_small_test_image()
