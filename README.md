# Remote-Sensing-Workflow-Automation
## Overview:
This project focuses on testing the efficiency and accuracy of 4 different combinations of remote sensing workflows. This project uses Machine Learning (ML) and Deep Learning (DL) methods to mask clouds and define land cover types in pixels in a satellite image acquired from Sentinel-2.
## How it works:

### 1. Preprocessing:
This script runs by accessing the STAC API through the requests library where it queries a result from the Microsoft Planetary Computer (MPC) based off of a bounding box for an image from Sentinel-2 Level 2A. After querying and acquiring access to the image, each band is individually downloaded into a folder where all processing will be done. The bands are then clipped to the user-specified extent and stacked into a single multispectral raster file at a 10 meter spatial resolution.
