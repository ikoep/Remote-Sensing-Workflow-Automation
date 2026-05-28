import arcpy # importing the arcpy library for geoprocessing
import os # importing the os library for file management
import importlib # importing importlib library to reload files
import time # used to track run times
from datetime import datetime, timedelta # for editing and rearraging esri arcgis dates into the microsoft planetary computer format

# all custom workflow scripts
import stac_test
import preprocess
import cloud_mask
import land_classification

# reloads files from RAM (reset)
importlib.reload(stac_test) 
importlib.reload(preprocess)
importlib.reload(cloud_mask)
importlib.reload(land_classification)

# grabs the parameters from the ArcGIS Pro script tool
downloads = arcpy.GetParameterAsText(0) # download location (Warning: this will use a lot of storage space!)
bbox_arcgis = arcpy.GetParameter(1) # this is the bounding box, the drawn or imported extent of the area of interest
wgs84 = arcpy.SpatialReference(4326)
extent = bbox_arcgis.projectAs(wgs84)
bbox = [extent.XMin, extent.YMin, extent.XMax, extent.YMax]

start_date = arcpy.GetParameterAsText(2).split(" ")[0] # start and end dates split to fit MPC date syntax requirements
end_date = arcpy.GetParameterAsText(3).split(" ")[0]
start_iso = datetime.strptime(start_date, "%m/%d/%Y").strftime("%Y-%m-%d") # fitted to ISO format
end_iso = datetime.strptime(end_date, "%m/%d/%Y").strftime("%Y-%m-%d")

cloud_cover = arcpy.GetParameterAsText(4) # as int
rf_model_file = arcpy.GetParameterAsText(5) # cloud mask training polygons file

rf_lc_model = arcpy.GetParameterAsText(6) # land cover training polygons file

# these act as on off boolean switches to have an interchangeable pipeline
RUN_DOWNLOADS = arcpy.GetParameter(7)       # Runs all proprocessing steps including bands downloads, clipping, and composite bands
RUN_CUSTOM_MASK = arcpy.GetParameter(8)     # RF Mask
RUN_DEEP_LEARNING = arcpy.GetParameter(9)   # DL Mask
RUN_RF_CLASSIFICATION = arcpy.GetParameter(10) # RF Classification
RUN_DL_CLASSIFICATION = arcpy.GetParameter(11) # DL Classification
# environment settings
arcpy.env.overwriteOutput = True # for running multiple attempts for testing(this can be toggled off in the script by setting it to False)
arcpy.env.cellSize = 10 # this forces the spatial resolution to be at 10 meters
arcpy.env.workspace = downloads # sets geodatabase to the downloads folder path
arcpy.env.randomGenerator = "1" # this forces a locked in seed so that each run is the same for scientific reproducability

# this is the time log for each function
time_log_path = os.path.join(downloads, "Analysis_Times_Log.txt") # connects to the downloads folder with the file name
with open(time_log_path, "a") as log: # as long as there is a run happening 
    log.write(f"\n\n--- NEW RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n") # print that there is a new run with the run times

# this adds all the layers when run to the map
aprx = arcpy.mp.ArcGISProject("CURRENT") 
current_map = aprx.activeMap

# this calculates the date ranges that will be split up so that the script can find good cloud cover images from the Microsoft Planetary Computer of the same place to mosaic
end_date_1 = datetime.strptime(end_iso, "%Y-%m-%d")
date_ranges = [
    f"{start_iso}/{end_iso}",
    f"{(end_date_1 + timedelta(days=1)).strftime('%Y-%m-%d')}/{(end_date_1 + timedelta(days=30)).strftime('%Y-%m-%d')}",
    f"{(end_date_1 + timedelta(days=31)).strftime('%Y-%m-%d')}/{(end_date_1 + timedelta(days=60)).strftime('%Y-%m-%d')}"
]

mosaic_to_classify = None # this is kept blank until it holds the cloud free mosaic that will be sent to the land cover classification

# PREPROCESSING AND CLOUD MASKING
start_mask_time = time.time()
swiss_cheese_rasters = [] # array to hold punched out cloud images

for date in date_ranges:
    arcpy.AddMessage(f"\n--- PROCESSING DATE RANGE: {date} ---")
    current_cc_limit = int(cloud_cover) if cloud_cover.isdigit() else 10 # looks for user specified cloud cover limit in the images
    result = None
    
    while current_cc_limit <= 100:
        result = stac_test.getExtentParameters(bbox, date, str(current_cc_limit))
        if result is not None and result[0] is not None:
            arcpy.AddMessage(f"Image found at {current_cc_limit}% cloud cover limit.")
            break # it found a good image
        else:
            current_cc_limit += 20 # it wasn't able to find a good image so it is bumped up by 20%
            
    if result is None or result[0] is None:
        arcpy.AddWarning(f"No valid images found for {date}. Skipping.") # if it doesn't find an image at 100% then SKIP

        continue
        
    image, image_id = result
    # file paths 
    clipped_multispectral = os.path.join(downloads, f"{image_id}_aoi.tif")
    clipped_scl = os.path.join(downloads, f"{image_id}_SCL_aoi.tif")
    #  DOWNLOAD BANDS
    if RUN_DOWNLOADS: 
        arcpy.AddMessage(f"Downloading and stacking {image_id}...")
        stac_test.downloadBands(image, image_id, downloads) # downloads the bands
        clipped_multispectral = preprocess.layerStack(image_id, downloads, bbox_arcgis) # stacks the bands into one image
        
        raw_scl = os.path.join(downloads, f"{image_id}_SCL.tif") 
        if arcpy.Exists(raw_scl):
            preprocess.clipLayer(raw_scl, bbox_arcgis, downloads, image_id)
    # CLOUD MASK
    if RUN_CUSTOM_MASK: # run random forest machine learning mask
        arcpy.AddMessage("Running Random Forest Masking...")
        if rf_model_file and arcpy.Exists(rf_model_file): # checks for an .ecd training polygons file
            rf_mask = cloud_mask.RandomForest_CM(clipped_multispectral, rf_model_file, downloads, image_id) # runs the random forest mask
            scl_mask = cloud_mask.trainingMask(clipped_scl, downloads, image_id) # uses the scl band as a composite mask
            
            if rf_mask and scl_mask:
                mega_mask = cloud_mask.combineMasks(rf_mask, scl_mask, downloads, image_id) # combines both scl and rf masks
                if mega_mask:
                    swiss_cheese = cloud_mask.applyCloudMask(clipped_multispectral, mega_mask, downloads, image_id, buffer_pixels=5) # applies the mask to the image
                    if swiss_cheese:
                        swiss_cheese_rasters.append(swiss_cheese)

    elif RUN_DEEP_LEARNING: # run deep learning mask
        arcpy.AddMessage("Running Deep Learning Masking...")
        dlpk_path = r"E:\Projects\CAPSTONE\Python_Tests\cloud_mask.dlpk" # finds dlpk file from path
        dl_mask = cloud_mask.DeepLearning_CM(clipped_multispectral, downloads, dlpk_path, image_id) # runs deep learning mask
        if dl_mask:
            swiss_cheese = cloud_mask.applyCloudMaskDL(clipped_multispectral, dl_mask, downloads, image_id, buffer_pixels=5) # applies the cloud mask
            if swiss_cheese:
                swiss_cheese_rasters.append(swiss_cheese)
    else:
        swiss_cheese_rasters.append(clipped_multispectral)

# mosaics the masked images together 
if len(swiss_cheese_rasters) > 1:
    arcpy.AddMessage("\n--- MOSAICKING CLOUD-FREE IMAGES TOGETHER ---")
    mosaic_name = "Capstone_Mosaic_RF.tif" if RUN_CUSTOM_MASK else "Capstone_Mosaic_DL.tif" # this gives only 2 options for names and forces the deep learning tag onto it if RUN_CUSTOM_MASK (RF) isn't run
    mosaic_to_classify = os.path.join(downloads, mosaic_name)
    
    try:
        arcpy.management.MosaicToNewRaster(
            input_rasters=swiss_cheese_rasters, 
            output_location=downloads, 
            raster_dataset_name_with_extension=mosaic_name, 
            coordinate_system_for_the_raster=arcpy.Describe(swiss_cheese_rasters[0]).spatialReference, 
            pixel_type="32_BIT_FLOAT", 
            cellsize=10,
            number_of_bands=12, 
            mosaic_method="FIRST" 
        ) # attempts to mosaic the images by first order
        arcpy.AddMessage("SUCCESS! Mosaic created.")
        current_map.addDataFromPath(mosaic_to_classify)
    except arcpy.ExecuteError:
        arcpy.AddError("Mosaic failed.")

# Log Masking Time
mask_time_elapsed = round(time.time() - start_mask_time, 2)
with open(time_log_path, "a") as log:
    log.write(f"Cloud Masking & Mosaicking Time: {mask_time_elapsed} seconds\n") # clouds cloud mask time to the file

# LAND COVER CLASSIFICATION
start_class_time = time.time()

if mosaic_to_classify and arcpy.Exists(mosaic_to_classify): # this section of the script won't run unless there is a valid masked and mosaicked image
    
    if RUN_RF_CLASSIFICATION:
        rf_ecd_path = r"E:\Projects\CAPSTONE\CM_Training_Poly\rf_ecd_lc.ecd" # for simple testing on my machine a file path suffices. Anyone looking to run this script will either need to replace the file path with the given training polygons or assign their own
        final_rf_map = land_classification.classifyPixelsRF(downloads, "FinalMap", mosaic_to_classify, rf_lc_model)
        
        if final_rf_map:
            land_classification.apply_5_class_colors(downloads, final_rf_map)
            current_map.addDataFromPath(final_rf_map)

    if RUN_DL_CLASSIFICATION:
        dl_model = r"E:\Projects\CAPSTONE\Python_Tests\corine_landcover.dlpk" # this is the file path to Esri's land cover classification deep learning model, again this path will need to be changed to run on another machine
        final_dl_map = land_classification.classifyPixelsDL(downloads, "FinalMap", mosaic_to_classify, dl_model)
        
        if final_dl_map:
            land_classification.apply_5_class_colors(downloads, final_dl_map)
            current_map.addDataFromPath(final_dl_map)

# Log Classification Time
class_time_elapsed = round(time.time() - start_class_time, 2)
with open(time_log_path, "a") as log:
    log.write(f"Classification Time: {class_time_elapsed} seconds\n")
    log.write(f"TOTAL WORKFLOW TIME: {mask_time_elapsed + class_time_elapsed} seconds\n")