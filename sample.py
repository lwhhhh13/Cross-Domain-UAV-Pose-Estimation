import os
import cv2
import glob
import numpy as np
from PIL import Image
from itertools import chain
import torch
from models.demo_superpoint import SuperPointFrontend

superpoint = SuperPointFrontend(
    weights_path= "",
    nms_dist=4,
    conf_thresh=0.1,
    nn_thresh=0.2,
    cuda=True  # If GPU support is available, it can be set to True.
)

def extract_features_superpoint(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    pts, desc, _ = superpoint.run(gray_image)
    desc = torch.tensor(desc).T
    confidences = pts[2]

    # Select keypoints with confidence greater than 0.2.
    valid_indices = [i for i, conf in enumerate(confidences) if conf > 0.2] # 0.2

    # valid_indices = [i for i, conf in enumerate(confidences) ]
    keypoints = [cv2.KeyPoint(x=pts[0, i], y=pts[1, i], size=1) for i in valid_indices]

    # Sort keypoints and descriptors based on confidence, and select the top 1024
    sorted_indices = sorted(valid_indices, key=lambda i: confidences[i], reverse=True)
    top_indices = sorted_indices[:1024]  # Select the top 1024 points with the highest confidence.


    keypoints = [keypoints[i] for i in top_indices]


    desc = desc[top_indices]
    return keypoints,desc
# get depth file
def get_depth_frames(scene):
    frames = []
    seqs = glob.glob(os.path.join(root, "the path to the dat files"))
    for seq in seqs:
       frames += sorted(glob.glob(os.path.join(seq, "*.dat")))
    # frames = os.path.join(seqs[0], f"{scene}.dat")
    return frames

# Compute the bounding box to calculate the coordinates of the 8 vertices of the cubic bounding box for the given point p.
def compute_bounding_box(p):
    return np.array(
        [
            [p[0] - radius, p[1] - radius, p[2] - radius],
            [p[0] - radius, p[1] - radius, p[2] + radius],
            [p[0] - radius, p[1] + radius, p[2] - radius],
            [p[0] - radius, p[1] + radius, p[2] + radius],
            [p[0] + radius, p[1] - radius, p[2] - radius],
            [p[0] + radius, p[1] - radius, p[2] + radius],
            [p[0] + radius, p[1] + radius, p[2] - radius],
            [p[0] + radius, p[1] + radius, p[2] + radius],
        ]
    )

# Extract point cloud data from the given depth map and color image.
def extract_point_cloud(depth, color, w, h, origin, K,T0):
    # w: A tuple containing two integers, representing the horizontal range of the selected region in the depth map and color image. w[0] is the column index of the left boundary, and w[1] is the column index of the right boundary (exclusive). That is, the selected region is [w[0], w[1]).
    # h: A tuple containing two integers, representing the vertical range of the selected region in the depth map and color image. h[0] is the row index of the upper boundary, and h[1] is the row index of the lower boundary (exclusive). That is, the selected region is [h[0], h[1]).
    # origin: A 3D coordinate representing the origin position in the point cloud coordinate system. This point is usually the position of a pixel selected in the depth map in the camera coordinate system, or a reference point.
    # K: The camera intrinsic matrix, a 3x3 matrix used to convert pixel coordinates to 3D coordinates in the camera coordinate system.
    cloud = []
    for v in range(h[0], h[1]):
        for u in range(w[0], w[1]):
            z = depth[v, u]
            x = (u - K[0, 2]) * z / K[0, 0]
            y = (v - K[1, 2]) * z / K[1, 1]
            r = color[v, u, 0] / 255.0
            g = color[v, u, 1] / 255.0
            b = color[v, u, 2] / 255.0
            if z <= 0.0:
                continue
            if abs(x - origin[0]) >= radius:
                continue
            if abs(y - origin[1]) >= radius:
                continue
            if abs(z - origin[2]) >= radius:
                continue

            cloud.append([x, y, z, r, g, b])

    # Subsample point cloud
    cloud = np.array(cloud, dtype=np.float32)
    if cloud.shape[0] < 512 :
        return None

    if cloud.shape[0] > cloud_size:
        indices = np.random.choice(cloud.shape[0], cloud_size, replace=False)
        cloud = cloud[indices, :]
        cloud[:, 0:3] = (cloud[:, 0:3] - origin) / radius
        return cloud
    else :
        indices = np.random.choice(cloud.shape[0], cloud_size, replace=True)
        cloud = cloud[indices, :]
        cloud[:, 0:3] = (cloud[:, 0:3] - origin) / radius
        return cloud


def get_random_point_within_annulus(u0, v0, r_min, r_max, img_width, img_height):

    theta = np.random.uniform(0, 2 * np.pi)

    r = np.random.uniform(r_min, r_max)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    u1 = u0 + dx
    v1 = v0 + dy

    u1 = np.clip(int(u1), 0, img_width - 1)
    v1 = np.clip(int(v1), 0, img_height - 1)

    return u1, v1


# Extract a specific region from the color image and resize it to the specified size.
def extract_color_patch(color, w, h):
    image = Image.fromarray(color[h[0] : h[1], w[0] : w[1]])
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def compute_overlap(patch0, patch1):
    # Calculate the overlapping region of image patches.
    x0_min, x0_max = np.min(patch0[:, 0]), np.max(patch0[:, 0])
    y0_min, y0_max = np.min(patch0[:, 1]), np.max(patch0[:, 1])

    x1_min, x1_max = np.min(patch1[:, 0]), np.max(patch1[:, 0])
    y1_min, y1_max = np.min(patch1[:, 1]), np.max(patch1[:, 1])

    overlap_x_min = max(x0_min, x1_min)
    overlap_x_max = min(x0_max, x1_max)
    overlap_y_min = max(y0_min, y1_min)
    overlap_y_max = min(y0_max, y1_max)

    if overlap_x_min >= overlap_x_max or overlap_y_min >= overlap_y_max:
        return 0.0

    overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
    total_area = (x0_max - x0_min) * (y0_max - y0_min) + (x1_max - x1_min) * (y1_max - y1_min) - overlap_area

    return overlap_area / total_area


# Sample matching pairs
def sample_matching_pairs(scene):
    patches = [] # Initialize an empty list to store matching pair information.
    frames = get_depth_frames(scene) # Get all depth frame file paths in the current scene.

    fx = (720 / 13.2) * 8.8
    fy = (480 / 8.8) * 8.8
    cx = 720 / 2
    cy = 480 / 2
    K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    # K = np.loadtxt(os.path.join(.txt")) #


    # Pick a random depth frame
    path = np.random.choice(frames)
    T0 = np.loadtxt(path.replace(".dat", ".txt"))
    depth = torch.load(path).numpy()
    color = np.array(Image.open(path.replace(".dat", ".png")))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    depth[depth > cutoff] = 0.0
    if np.isnan(np.sum(T0)):
        return None


    keypoints, _ = extract_features_superpoint(color)
    kp1 = np.random.choice(keypoints)
    u0, v0 = int(kp1.pt[0]), int(kp1.pt[1])

    if depth[v0, u0] <= 0.0:
        return None

    # Compute bounding box
    z0 = depth[v0, u0]
    x0 = (u0 - K[0, 2]) * z0 / K[0, 0]
    y0 = (v0 - K[1, 2]) * z0 / K[1, 1]
    p0 = np.array([x0, y0, z0])
    q0 = np.matmul(T0[0:3, 0:3], p0) + T0[0:3, 3]
    b0 = compute_bounding_box(p0)

    b0[:, 0] = np.round(b0[:, 0] * K[0, 0] / b0[:, 2] + K[0, 2])
    b0[:, 1] = np.round(b0[:, 1] * K[1, 1] / b0[:, 2] + K[1, 2])

    # Get the depth patch
    x0 = np.array([np.min(b0[:, 0]), np.max(b0[:, 0])], dtype=np.int32)
    y0 = np.array([np.min(b0[:, 1]), np.max(b0[:, 1])], dtype=np.int32)
    if np.any(x0 < 0) or np.any(x0 >= depth.shape[1]):
        return None
    if np.any(y0 < 0) or np.any(y0 >= depth.shape[0]):
        return None

    patch0 = {}
    patch0["cloud"] = extract_point_cloud(depth, color, x0, y0, p0, K,T0)
    if patch0["cloud"] is None:
        patch0["color"] = None
    else:
        patch0["color"] = extract_color_patch(color, x0, y0)

    u1, v1 = get_random_point_within_annulus(u0, v0, 30, 50, 720, 480)
    u1 = np.random.choice(depth.shape[1])
    v1 = np.random.choice(depth.shape[0])


    if depth[v1, u1] <= 0.0:
        return None
    z1 = depth[v1, u1]
    x1 = (u1 - K[0, 2]) * z1 / K[0, 0]
    y1 = (v1 - K[1, 2]) * z1 / K[1, 1]
    p1 = np.array([x1, y1, z1])

    b1 = compute_bounding_box(p1)

    b1[:, 0] = np.round(b1[:, 0] * K[0, 0] / b1[:, 2] + K[0, 2])
    b1[:, 1] = np.round(b1[:, 1] * K[1, 1] / b1[:, 2] + K[1, 2])

    # Get the depth patch
    x1 = np.array([np.min(b1[:, 0]), np.max(b1[:, 0])], dtype=np.int32)
    y1 = np.array([np.min(b1[:, 1]), np.max(b1[:, 1])], dtype=np.int32)
    if np.any(x1 < 0) or np.any(x1 >= depth.shape[1]):
        return None
    if np.any(y1 < 0) or np.any(y1 >= depth.shape[0]):
        return None

    patch1 = {}
    patch1["cloud"] = extract_point_cloud(depth, color, x1, y1, p1, K,T0)
    if patch1["cloud"] is None:
        patch1["color"] = None
    else:
        patch1["color"] = extract_color_patch(color, x1, y1)
    sample_pairs = []

    if patch0["cloud"] is not None and patch1["cloud"] is not None:

        overlap_ratio = compute_overlap(patch0["color"], patch1["color"])

        sample_pairs.append({
            "pair_type": "positive",
            "color": patch0["color"],
            "cloud": patch0["cloud"],
            "overlap_ratio": 1.0
        })

        # #
        # sample_pairs.append({
        #     "pair_type": "positive",
        #     "color": patch1["color"],
        #     "cloud": patch1["cloud"],
        #     "overlap_ratio": 1.0
        # })

        # Determine positive and negative samples based on the overlap ratio.
        if overlap_ratio < 0.3:
            sample_pairs.append({
                "pair_type": "negative",
                "color": patch0["color"],
                "cloud": patch1["cloud"],
                "overlap_ratio": overlap_ratio
            })
            sample_pairs.append({
                "pair_type": "negative",
                "color": patch1["color"],
                "cloud": patch0["cloud"],
                "overlap_ratio": overlap_ratio
            })
        elif overlap_ratio > 0.7:
            sample_pairs.append({
                "pair_type": "positive",
                "color": patch0["color"],
                "cloud": patch1["cloud"],
                "overlap_ratio": overlap_ratio
            })
            sample_pairs.append({
                "pair_type": "positive",
                "color": patch1["color"],
                "cloud": patch0["cloud"],
                "overlap_ratio": overlap_ratio
            })

    # elif patch0["cloud"] is not None or patch1["cloud"] is not None:
    #     if patch0["cloud"] is not None:
    #         sample_pairs.append({
    #             "pair_type": "positive",
    #             "color": patch0["color"],
    #             "cloud": patch0["cloud"],
    #             "overlap_ratio": 1.0
    #         })
    #     elif patch1["cloud"] is not None:
    #         #
    #         sample_pairs.append({
    #             "pair_type": "positive",
    #             "color": patch1["color"],
    #             "cloud": patch1["cloud"],
    #             "overlap_ratio": 1.0
    #         })

    if len(sample_pairs) <= 1: # Return None if there are not enough matching samples.
        return None
    return sample_pairs # Return all matching point clouds and image patches.


def save_batch_npy(fname, batch):
    patches = list(chain(*batch))

    valid_patches = [patch for patch in patches if patch["cloud"] is not None]

    if len(valid_patches) == 0:
        print("No valid point clouds to save")
        return

    # Ensure that the shape of all point cloud data and image data is consistent.
    cloud_shapes = set(patch["cloud"].shape for patch in valid_patches)
    image_shapes = set(patch["color"].shape for patch in valid_patches)
    if len(cloud_shapes) > 1:
        raise ValueError("All point clouds must have the same shape")
    if len(image_shapes) > 1:
        raise ValueError("All images must have the same shape")

    # Calculate the length of each sample.
    indices = [len(sample) for sample in batch]
    indices = np.array(indices, dtype=np.int32)
    # Create a label mapping dictionary.
    label_map = {
        'positive': 1,
        'negative': 0
    }

    clouds = np.stack([patch["cloud"] for patch in valid_patches])
    images = np.stack([patch["color"] for patch in valid_patches])
    labels = np.array([label_map[patch["pair_type"]] for patch in valid_patches], dtype=np.int32)
    overlap = np.stack([patch["overlap_ratio"] for patch in valid_patches])
    positive_count = sum(1 for patch in valid_patches if patch["pair_type"] == "positive")
    negative_count = sum(1 for patch in valid_patches if patch["pair_type"] == "negative")
    print(positive_count,negative_count)

    # save .npy file
    np.savez(fname, clouds=clouds, images=images, indices=indices,labels=labels,overlap = overlap)

# Get all filenames in the folder
folder_path = ""
scenes = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

# Set the root directory
root = ""

dataset_size =  # The total size of the dataset, specifying the total number of samples to be processed.
batch_size =  # Batch size for each processing.
radius =  # Radius used in the point cloud extraction process for calculating the bounding box of the point cloud.
cutoff =  # Upper limit of the depth value, depths exceeding this value are ignored
cloud_size =  # Number of points in each point cloud.
image_size =  # Size of the extracted color image patches.
seed =  # Random seed.

size =  # Initialize the sample counter.
batch = [] # Initialize the batch list.
while size < dataset_size:
    scene = np.random.choice(scenes) # Randomly select a scene.
    sample = sample_matching_pairs(scene)  # Sample matching pairs from the scene.
    if sample is None:  # Skip if the sample is empty. A sample contains two sample pairs.
        continue
    size += 1
    batch += [sample]
    a = len(batch)
    print("Sample matching patches [{}/{}]".format(size, dataset_size))

    # Save batch if needed
    if len(batch) == batch_size: # Save the data if the length of the batch list equals the batch size.
        i = size // batch_size # Calculate the index of the current batch.

        fname = "".format(i) # Generate the file path fname for saving the data as an npz file.
        save_batch_npy(fname, batch)
        print("> Saving batch to {}...".format(fname))
        batch = []
