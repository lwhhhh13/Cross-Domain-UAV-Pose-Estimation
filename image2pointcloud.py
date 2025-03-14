import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

def main():
    parser = argparse.ArgumentParser(description='Process images to generate point clouds.')
    parser.add_argument('--load-from', type=str, default="Path to the depth images", help='Path to the depth images.')
    parser.add_argument('--max-real-depth', default= ,type=float,  help='Maximum real-depth value.')
    parser.add_argument('--img-path', type=str, default="Path to the input images", help='Path to the input images.')
    parser.add_argument('--outdir', type=str, default= "Directory to save output point clouds", help='Directory to save output point clouds.')
    parser.add_argument('--focal-length-x', type=float, default= ,help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', type=float, default= ,help='Focal length along the y-axis.')

    args = parser.parse_args()


    image_folder = args.img_path
    depth_folder = args.load_from
    output_folder = args.outdir


    image_filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
    depth_filenames = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith('.dat')])


    os.makedirs(output_folder, exist_ok=True)


    for k, (image_filename, depth_filename) in enumerate(zip(image_filenames, depth_filenames)):
        print(f'Processing {k + 1}/{len(image_filenames)}: {image_filename}')


        color_image = Image.open(image_filename).convert('RGB')
        width, height = color_image.size


        pred = torch.load(depth_filename).numpy()


        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)


        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x
        y = (y - height / 2) / args.focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

        colors = np.array(color_image).reshape(-1, 3) / 255.0


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        output_filename = os.path.join(output_folder, os.path.splitext(os.path.basename(image_filename))[0] + ".ply")
        o3d.io.write_point_cloud(output_filename, pcd)

if __name__ == '__main__':
    main()
