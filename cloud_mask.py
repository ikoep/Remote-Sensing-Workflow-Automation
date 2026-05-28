import arcpy
import os
from arcpy.sa import * # imports all methods specifically from the spatial analysis library
# RANDOM FOREST MACHINE LEARNING CLOUD MASK
def trainingMask(clipped_scl_band, downloads_folder, image_id): # this is the Sen2Cor SCL band that is used to help fill the gaps from the machine learning
    arcpy.CheckOutExtension("Spatial")
    training_mask_output = os.path.join(downloads_folder, f"{image_id}_Binary_from_SCL_Training_Mask.tif")
    arcpy.sa.Mask(clipped_scl_band, no_data_values=6, included_ranges=[3,9])
    
    try:
        scl_ranges = arcpy.sa.RemapValue([[0,2], [1,2], [2,2], [3,2], [4,0], [5,0], [6,0], [7,0], [8,1], [9,1], [10,1], [11,1]]) # grabbing scl categories and reassigning them to clear, cloud, and shadow
        training_mask = arcpy.sa.Reclassify(clipped_scl_band, "Value", scl_ranges) # reclassifies by value column
        training_mask.save(training_mask_output)
        return training_mask_output
    except arcpy.ExecuteError:
        return None
    finally:
        arcpy.CheckInExtension("Spatial")

def RandomForest_CM(multispectral_image_to_mask, ecd_file, downloads, image_id): # the random forest algorithm run on the predefined seed and trained polygons identify between clear, cloud, and shadow
    arcpy.CheckOutExtension("Spatial")
    final_masked_image = os.path.join(downloads, f"{image_id}_final_CM.tif")
    try:
        cm_raster = arcpy.sa.ClassifyRaster(multispectral_image_to_mask, ecd_file) # attempts to classify the multispectral image with the polygons
        cm_raster.save(final_masked_image)
        return final_masked_image
    except arcpy.ExecuteError:
        arcpy.AddError("\n--- RANDOM FOREST CLASSIFICATION FAILED ---")
        arcpy.AddError(arcpy.GetMessages(2))
        return None
    except Exception as e:
        arcpy.AddError(f"\n--- PYTHON CRASHED IN RF_CM ---\n{str(e)}")
        return None
    finally:
        arcpy.CheckInExtension("Spatial")

def combineMasks(rf_mask, scl_mask, downloads, image_id): # adds SCL and regular mask together
    arcpy.CheckOutExtension("Spatial")
    ultimate_mega_mask = os.path.join(downloads, f"{image_id}_ultimate_mega_mask.tif")
    try:
        combo_raster = Con((Raster(rf_mask) == 0) & (Raster(scl_mask) == 2), 0, Raster(scl_mask))
        combo_raster.save(ultimate_mega_mask)
        return ultimate_mega_mask
    except arcpy.ExecuteError:
        return None
    finally:
        arcpy.CheckInExtension("Spatial")

def applyCloudMask(multispectral, mask, downloads, image_id, buffer_pixels=5): # this applies the mask to the actual image and cuts out any clouds or shadows it could find
    arcpy.CheckOutExtension("Spatial")
    swiss_cheese_output = os.path.join(downloads, f"{image_id}_RF_CM.tif")

    try:
        arcpy.AddMessage(f"Expanding RF cloud mask by {buffer_pixels} pixels to capture edges...") # this mostly handles annoying thin white edges
        
        expanded_mask = arcpy.sa.Expand(Raster(mask), buffer_pixels, [1, 2]) # instructing the mask to only buffer this for clouds and shadows
        
        valid_mask = arcpy.sa.SetNull(expanded_mask > 0, 1) # assigns clouds and shadows to NoData and clear to 1
        
        arcpy.AddMessage("Extracting clear pixels and punching widened holes...")

        clean_raster = arcpy.sa.ExtractByMask(multispectral, valid_mask) # punches the actual holes out of the image
        clean_raster.save(swiss_cheese_output)

        arcpy.AddMessage(f"Success! {image_id} has been cut with a {buffer_pixels}-pixel buffer.")
        return swiss_cheese_output
        
    except Exception as e:
        arcpy.AddError(f"Failed to apply RF mask: {str(e)}")
        return None
    finally:
        arcpy.CheckInExtension("Spatial")
# ESRI DEEP LEARNING CLOUD MASK 
def DeepLearning_CM(multispectral_image, downloads, pretrained_cm, image_id):
    # CRITICAL: Check out the extension and force GPU usage
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.env.processorType = "GPU"
    
    output_dl_cm = os.path.join(downloads, f"{image_id}_Output_DL_CM.tif")
    
    try:
        # Run the tool and capture the output as a variable
        dl_raster = arcpy.ia.ClassifyPixelsUsingDeepLearning(
            in_raster=multispectral_image, 
            in_model_definition=pretrained_cm,
            processing_mode="PROCESS_AS_MOSAICKED_IMAGE"
        )
        
        dl_raster.save(output_dl_cm)
        arcpy.AddMessage(f"Successfully saved {image_id}_Output_DL_CM.tif")
        return output_dl_cm
        
    except Exception as e:
        arcpy.AddError(f"Deep Learning failed on {image_id}: {str(e)}")
        
    finally:
        arcpy.CheckInExtension("ImageAnalyst")
        arcpy.env.processorType = "CPU" # reset

def applyCloudMaskDL(multispectral_image, output_dl_cm, downloads, image_id, buffer_pixels=5):
    arcpy.CheckOutExtension("Spatial")
    dl_cloud_free = os.path.join(downloads, f"{image_id}_DL_CM.tif")

    try:
        arcpy.AddMessage(f"Expanding cloud mask by {buffer_pixels} pixels to capture edges/shadows...")
        dl_raster = Raster(output_dl_cm)
        
        expanded_clouds = arcpy.sa.Expand(dl_raster, buffer_pixels, [1, 2, 3]) # 5 pixel buffer
        
        arcpy.AddMessage("Isolating clear pixels from expanded mask...")

        clean_mask = Con(IsNull(expanded_clouds), 1, Con(expanded_clouds == 0, 1)) # NoData and Clear 1
        
        arcpy.AddMessage("Extracting clear pixels and punching widened holes...")
        cut = ExtractByMask(multispectral_image, clean_mask)
        cut.save(dl_cloud_free)
        
        arcpy.AddMessage(f"Success! {image_id} has been cut with a {buffer_pixels}-pixel buffer.")
        return dl_cloud_free
        
    except arcpy.ExecuteError:
        arcpy.AddError("Failed to apply DL mask")
        arcpy.AddError(arcpy.GetMessages(2))
        return None
    except Exception as e:
        arcpy.AddError(f"Python error: {str(e)}")
        return None
    finally:
        arcpy.CheckInExtension("Spatial")