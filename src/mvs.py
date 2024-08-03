import numpy as np
import cv2

def generate_point_cloud(color_image, depth_image, fx, fy, cx, cy):
    """
    Generate a point cloud from color and depth images.

    Args:
        color_image (np.array): The color image.
        depth_image (np.array): The depth image.
        fx (float): Focal length x.
        fy (float): Focal length y.
        cx (float): Principal point x.
        cy (float): Principal point y.

    Returns:
        np.array: The generated point cloud.
    """
    h, w = depth_image.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth_image / 1000.0
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points
