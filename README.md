# Cross-Domain UAV Pose Estimation: A Novel Attempt in UAV Visual Localization
We present a novel attempt in UAV visual localization by leveraging cross-domain image-point cloud matching and we also introduce two new datasets.

Our training and testing code is still being organized, and we will release it as soon as possible.
## Data Download
Our datasets can be accessed using the following method:

**Baidu Netdisk**: https://pan.baidu.com/s/17yfB-p-9kE_LTQuo4aPP7Q (Access Code: gcjf)
 

We also provide several scripts for processing the dataset.

If you want to generate a point cloud from images, you can run the following script:
```bash
python image2pointcloud.py
```

If you want to collect 2D-3D matching pairs with both positive and negative samples, you can use the following script:
```bash
python sample.py
```
