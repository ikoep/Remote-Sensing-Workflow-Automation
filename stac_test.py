import requests # allows us to use pystac_client library that connects to the Microsoft Planetary Computer (MPC)
import arcpy
import os

def getExtentParameters(bbox_arcgis, date_range, cloud_cover):
    search_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search" # url to connect to the search
    search_parameters = {"collections": ["sentinel-2-l2a"], "bbox": bbox_arcgis, "datetime": date_range, "query": {"eo:cloud_cover": {"lt": int(cloud_cover)}}} # these are the parameters that are set: the collection of imagery is from sentinel-2-l2a, extent set to the bbox, date time set to the user defined range, and the query specifications of cloud cover
    
    try:
        response = requests.post(search_url, json=search_parameters) # attempts to connect with a json request
        response.raise_for_status()
    except Exception as e:
        arcpy.AddError(f"STAC API Request failed. {e}")
        return None, None

    search_items = response.json().get('features', [])
    if not search_items:
        return None, None

    best_result = search_items[0] # at index 0 best results
    return best_result, best_result['id'] # returns results with id

def downloadBands(image, image_id, downloads_folder):
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"] # list of bands to download
    url_key = "https://planetarycomputer.microsoft.com/api/sas/v1/sign" # key to access the download

    for band in bands:
        file_path = os.path.join(downloads_folder, f"{image_id}_{band}.tif")

        # resume but skip if the file already exists
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000000: # as long as the folder to download the bands and the size of the band isn't enormous execute the following code
            arcpy.AddMessage(f"  -> {band} is already fully downloaded. Skipping!")
            continue

        arcpy.AddMessage(f"  -> Downloading {band}...")
        raw_band = image['assets'][band]['href']
        
        try: # attemps to download bands with access
            response = requests.get(url_key, params={"href": raw_band})
            url = response.json().get("href")
            
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): # downloads in 8192 mb in order to not overload RAM
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            arcpy.AddError(f"Download failed on {band}. Error: {str(e)}")