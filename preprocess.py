import arcpy
import os

def layerStack(image_id, downloads, bbox_arcgis):
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True
    
    sample_band = os.path.join(downloads, f"{image_id}_B02.tif")
    raster_sr = arcpy.Describe(sample_band).spatialReference # grabs band as a sample and reads its coordinate system (cr)
    
    bbox_projected = bbox_arcgis.projectAs(raster_sr) # projects the bounding box to the sample cs
    extent_fix = f"{bbox_projected.XMin} {bbox_projected.YMin} {bbox_projected.XMax} {bbox_projected.YMax}" # fixes extent to their set properties
    
    # All 12 Sentinel-2 spectral bands
    bands_to_process = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    processed_bands = [] # once downloaded and processed each band is added to this list

    arcpy.AddMessage("Clipping and Resampling multispectral bands to 10m...")
    for band in bands_to_process:
        raw_band = os.path.join(downloads, f"{image_id}_{band}.tif")
        temp_clip = os.path.join(downloads, f"temp_{image_id}_{band}.tif")
        clipped_resampled_band = os.path.join(downloads, f"{image_id}_{band}_aoi.tif")
        
        if not arcpy.Exists(clipped_resampled_band): # raw unclipped bands will be run through the following code
            arcpy.management.Clip(in_raster=raw_band, rectangle=extent_fix, out_raster=temp_clip) # clipped to proper extent boundary
            
            arcpy.management.Resample(temp_clip, clipped_resampled_band, "10 10", "BILINEAR") # turns band into 10m spatial resolution at a bilinear light reflectance
            arcpy.management.Delete(temp_clip)

        processed_bands.append(clipped_resampled_band) # once processed, it goes into the finished list

    output_stack = os.path.join(downloads, f"{image_id}_aoi.tif")
    arcpy.management.CompositeBands(processed_bands, output_stack) # all bands are then stacked together to form a single image
    
    arcpy.CheckInExtension("Spatial")
    return output_stack

def clipLayer(raw_scl_path, bbox_arcgis, downloads, image_id): # SCL
    raster_sr = arcpy.Describe(raw_scl_path).spatialReference 
    bbox_projected = bbox_arcgis.projectAs(raster_sr)
    extent_fix = f"{bbox_projected.XMin} {bbox_projected.YMin} {bbox_projected.XMax} {bbox_projected.YMax}"
    
    clipped_scl_band = os.path.join(downloads, f"{image_id}_SCL_aoi.tif") 

    arcpy.AddMessage("Clipping and Resampling SCL categorical band...")
    temp_clip = os.path.join(downloads, f"temp_{image_id}_SCL.tif")
    arcpy.management.Clip(in_raster=raw_scl_path, rectangle=extent_fix, out_raster=temp_clip) # clip SCL
 
    arcpy.management.Resample(temp_clip, clipped_scl_band, "10 10", "NEAREST") # SCL to 10m pixels
    arcpy.management.Delete(temp_clip)

    return clipped_scl_band