import arcpy
import os


# RANDOM FOREST 
def classifyPixelsRF(downloads, image_id, mosaic_mask, ecd_path):
    arcpy.CheckOutExtension("Spatial")
    

    rf_lclass = os.path.join(downloads, f"{image_id}_rf_classified.crf") # force .crf as it is the default, it can be turned into a tif later

    try:
        arcpy.AddMessage(f"Running Random Forest Land Cover Classification for {image_id}...")
        
        # runs the classification as a temporary raster in RAM
        classified_raster = arcpy.sa.ClassifyRaster(
            in_raster=mosaic_mask, 
            in_classifier_definition=ecd_path
        )
        
        classified_raster.save(rf_lclass)
        
        arcpy.AddMessage(f"Success! Land Cover map saved to {rf_lclass}")
        return rf_lclass
        
    except Exception as e:
        arcpy.AddError(f"Classification failed: {str(e)}")
        return None
        
    finally:
        arcpy.CheckInExtension("Spatial")


def classifyPixelsDL(downloads, image_id, mosaic_image, dlpk_path):
    arcpy.CheckOutExtension("ImageAnalyst")
    arcpy.CheckOutExtension("Spatial")
    
    original_processor = arcpy.env.processorType
    arcpy.env.processorType = "GPU" # forces GPU for DL tasks
    
    raw_dl_output = os.path.join(downloads, f"{image_id}_DL_Raw.crf")
    final_reclassed = os.path.join(downloads, f"{image_id}_DL_Final_5Class.tif")
    
    try:
        arcpy.AddMessage("\n--- Running Deep Learning Classification (GPU Enabled) ---")
        
        # deep learning land cover classification
        dl_result = arcpy.ia.ClassifyPixelsUsingDeepLearning(
            in_raster=mosaic_image,
            in_model_definition=dlpk_path,
            processing_mode="PROCESS_AS_MOSAICKED_IMAGE"
        )
        dl_result.save(raw_dl_output)
        arcpy.AddMessage("Raw DL classification complete. Starting Reclassification...")
        
        # thematic crosswalking into 5 classes
        remap_list = [
            [11, 3],  # Urban fabric -> Urban (3)
            [12, 3],  # Industrial/Commercial -> Urban (3)
            [13, 3],  # Mine dump/Construction -> Urban (3)
            [14, 3],  # Artificial non-ag veg -> Urban (3)
            [21, 4],  # Arable land -> Ag (4)
            [22, 4],  # Permanent crops -> Ag (4)
            [23, 4],  # Pastures -> Ag (4)
            [24, 4],  # Heterogeneous ag -> Ag (4)
            [31, 2],  # Forests -> Forest (2)
            [32, 5], # Scrub/Herbaceous -> Rangeland (5)
            [33, 5], # Open spaces -> Rangeland (5)
            [41, 1], # Wetlands -> Water (1)
            [51, 1],  # Waters -> Water (1)
            [99, "NODATA"]
        ]
        
        # converts to RemapValue object so it can be reclassified
        remap_values = arcpy.sa.RemapValue(remap_list)
        
        # raster is reclassified
        reclassed_raster = arcpy.sa.Reclassify(
            in_raster=raw_dl_output, 
            reclass_field="Value", 
            remap=remap_values, 
            missing_values="NODATA" # Ensures unmapped values become NoData
        )
        

        reclassed_raster.save(final_reclassed)
        arcpy.AddMessage(f"SUCCESS! Consolidated DL map saved to {final_reclassed}")
        
        return final_reclassed
        
    except arcpy.ExecuteError:
        arcpy.AddError("DL Classification or Reclassification failed.")
        arcpy.AddError(arcpy.GetMessages(2))
        return None
    except Exception as e:
        arcpy.AddError(f"Python error: {str(e)}")
        return None
    finally:
        # 3. Clean up and release licenses
        arcpy.env.processorType = original_processor # Sets it back to CPU for other tools
        arcpy.CheckInExtension("ImageAnalyst")
        arcpy.CheckInExtension("Spatial")

def apply_5_class_colors(downloads, raster_path):
    # prevents symbology crash
    try:
        arcpy.AddMessage("Note: Automated coloring bypassed. Map is saved and ready for the matrix!")
        return True
    except Exception as e:
        return False