import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label

def extract_regions_from_mask(rgb: torch.Tensor, mask: torch.Tensor, output_size = (15,15)):
    mask_np = mask
    mask_bin = mask_np > 0
    labeled_mask, num_features = label(mask_bin)

    results = []
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_mask == region_id)
        indices = np.argwhere(region_mask)
        if indices.shape[0] == 0:
            continue

        y_min, x_min = indices.min(0)
        y_max, x_max = indices.max(0)

        cropped = rgb[:, y_min:y_max+1, x_min:x_max+1] * torch.tensor(region_mask[y_min:y_max+1, x_min:x_max+1], dtype=rgb.dtype, device = rgb.device).unsqueeze(0)

        resized = F.interpolate(cropped.unsqueeze(0), size = output_size,mode = 'bilinear',align_corners = False)
        results.append(resized.squeeze(0))

    return results

rgb = torch.randn(3,128,128)
mask = torch.randint(0,5,(128,128))

patches = extract_regions_from_mask(rgb,mask)

print(f"region{len(patches)}, shape is {patches[0].shape}")