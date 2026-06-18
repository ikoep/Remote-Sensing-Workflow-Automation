# Remote-Sensing-Workflow-Automation
## Overview:
This project focuses on testing the efficiency and accuracy of 4 different combinations of remote sensing workflows. This project uses Machine Learning (ML) and Deep Learning (DL) methods to mask clouds and define land cover types in pixels in a satellite image acquired from Sentinel-2.
## How it works:

### 1. Preprocessing:
This script runs by accessing the STAC API through the requests library where it queries a result from the Microsoft Planetary Computer (MPC) based off of a bounding box for an image from Sentinel-2 Level 2A. After querying and acquiring access to the image, each band is individually downloaded into a folder where all processing will be done. The bands are then clipped to the user-specified extent and stacked into a single multispectral raster file at a 10 meter spatial resolution.

### 2. Cloud Masking:
This script operates in conjunction with the ArcGIS Pro script tool; therefore, the cloud mask and land cover features are toggleable. That being said, there are two kinds of cloud masks that are being used here. The first is a machine learning cloud mask that uses the scene classification layer and a training sample on the random forest algorithm to create a mask. It then does this three times on each image and then splices them together into a single raster. The second is a deep learning cloud mask that runs on Esri's pretrained deep learning model and runs through the same three images and appends them. 

### 3. Land Cover Classification:
