import numpy as np
import imageio.v2 as imageio

d = np.load("background.npy", allow_pickle=True).item()
rgb = d["rgb"]
print(rgb.dtype, rgb.shape, rgb.min(), rgb.max())

# Many pipelines store images as BGR; swap to RGB before saving.
rgb_to_save = rgb[..., ::-1]

depth = d["depth_raw"]
depth_min = float(depth.min())
depth_max = float(depth.max())
if depth_max > depth_min:
    depth_norm = (depth - depth_min) / (depth_max - depth_min)
else:
    depth_norm = np.zeros_like(depth)
depth_u16 = (depth_norm * 65535.0).round().astype(np.uint16)

imageio.imwrite("rgb.png", rgb_to_save)
imageio.imwrite("depth.png", depth_u16)
