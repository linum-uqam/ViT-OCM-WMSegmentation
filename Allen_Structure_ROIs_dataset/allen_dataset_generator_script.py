

from allensdk.api.queries.image_download_api import ImageDownloadApi
from allensdk.api.queries.synchronization_api import SynchronizationApi
from allensdk.config.manifest import Manifest
from allensdk.core.reference_space_cache import ReferenceSpaceCache
import pandas as pd
import os
import argparse



def getROIByIndex(section_image_id,x,y,width,height,destination_path,image_api):
    """ This function gets the ROI from the Allen Brain section image and saves it in the destination_path. 
    Parameters:
    -----------
    section_image_id: int
        The id of the section image to get the ROI from.
    x: int
        The x coordinate of the ROI.
    y: int
        The y coordinate of the ROI.
    width: int
        The width of the ROI.
    height: int
        The height of the ROI.
    destination_path: str
        The path to save the ROI.
    """

    image_api.download_section_image(section_image_id, destination_path, top = y, left = x, width = width, height = height,downsample_dimensions  = False,downsample = 1)

def get_reference_to_image(x,y,z,section_data_set_ids,reference_space_id,structure_acronym,structure_id,width,height,image_api,sync_api):
    """ This function gets the image from a reference space coordination for a list of section data sets.
    Parameters:
    -----------
    x: int 
        The x coordinate in the reference space.
    y: int
        The y coordinate in the reference space.
    z: int
        The z coordinate in the reference space.
    section_data_set_ids: list of int
        The list of section data set ids to get the Rois from.
    reference_space_id: int
        The id of the reference space 9 is for coronal data, 10 is for sagittal data.
    structure_acronym: str
        The acronym of the structure.
    structure_id: int
        The id of the structure.
    width: int
        The width of the ROI.
    height: int
        The height of the ROI.
    Returns:
    --------
    None.
    """
    # get the section image id and the x, y coordinates of the point in the reference space for the specific strucure data set.
    json_response  = sync_api.get_reference_to_image(x=x, y=  y, z= z, section_data_set_ids=section_data_set_ids,reference_space_id=reference_space_id)
    format_str = '.jpg'
    overlap = False 
    # to check if the new ROI overlap with an old one. if there is less then 100 distance on the x and y coords we skip it.
    # the api call will return the closet available coordinates and section image in the dataset. Sometimes the thickness of the dataset or section is high. so the 
    # api will return the closest the same point more than one. we prevent this by setting a threshold on distance between the points of the same section image.
    for row in json_response:
        response_body = row['image_sync']
        for item in rows_list:
            if item['section_image_id'] == response_body['section_image_id'] and ( (abs(item['x_sec'] - response_body['x']) < 100) and (abs(item['y_sec'] - response_body['y']) < 100)):
                print("Warning: overlaped images will not be saved" )   
                overlap = True
                #print(overlap)
                break
        if overlap == True:     
            overlap == False
            continue
        
        directory_name = structure_acronym +'_'+ str(structure_id) # name of the directory acronym + structure id
        file_name = f"si-{response_body['section_image_id']}_x-{response_body['x']}_y-{response_body['y']}{format_str}"
        destination_path = os.path.join(directory_name, file_name)
        print(destination_path)
        Manifest.safe_make_parent_dirs(destination_path) 
        getROIByIndex(response_body['section_image_id'],
                        response_body['x'],
                        response_body['y'],
                        width,
                        height,
                        destination_path,
                        image_api = image_api,
                        )
        #saving the ROI information into a list. so we can create a csvs file later.
        dict1 = { 'section_data_set_id': response_body['section_data_set_id'],
                    'section_image_id': response_body['section_image_id'],
                    'x_sec':response_body['x'],
                    'y_sec':response_body['y'],
                    'structure_acronym':structure_acronym,
                    'structure_id':structure_id,
                    'x_ref':x,
                    'y_ref':y,
                    'z_ref':z,
                    'destination':destination_path}
        rows_list.append(dict1)
    # saving the list of ROI information into a csv file.
    df = pd.DataFrame(rows_list)
    df.to_csv('ROIs_description.csv', sep='\t', encoding='utf-8')

def getROIsFrom3DMasks_per_structure(mask,section_data_set_ids,structure_acronym,structure_id,reference_space_id,resolution,width,height,image_api,sync_api):
    """ This function gets the ROIs from a 3D mask for a list of section data sets.
    Parameters:
    -----------
    mask: numpy array
        The 3D mask of the structure. 
    section_data_set_ids: list of int
        The list of section data set ids to get the Rois from.
    structure_acronym: str
        The acronym of the structure.
    structure_id: int
        The id of the structure.
    reference_space_id: int
        The id of the reference space 9 is for coronal data, 10 is for sagittal data.
    resolution: int
        The resolution of the reference space that is being used (10, 25, 50, 100).
    width: int
        The width of the ROI.
    height: int
        The height of the ROI.
    Returns:
    --------
        count of the ROIS for the specific structure
    """
    #We use downsampling of size 1 when we acquire the images, to get bigger images with more information. Therefore, we need to double the size of the steps we are making 
    #to avoid overlapping between the resulting Rois. 
    w = int(width*2 / resolution)
    h = int(height*2 / resolution)
    steps_on_z_axis = 4 # steps to skip layers on the z axis. otherwise we will make a lot of uneccessary calls. 4 for the dataset of 20 thickness. 
    # and 8 for the dataset of 25 thickness.
    count = 0
    (mask_depth,mask_width, mask_height) = mask.shape
    print(mask.shape)
    for k in range(int(mask_depth/steps_on_z_axis)):
        for i in range(int(mask_width / w)):
            for j in range(int(mask_height / h)):
                if mask[k*steps_on_z_axis][i*w][j*h] > 0:
                    get_reference_to_image(z=j*h*resolution-width/2,
                                            y=i*w*resolution-height/2,
                                            x= k*25*steps_on_z_axis,
                                            section_data_set_ids=section_data_set_ids,
                                            reference_space_id=reference_space_id,
                                            structure_acronym=structure_acronym,
                                            structure_id=structure_id,
                                            width=width,
                                            height=height,
                                            image_api = image_api,
                                            sync_api = sync_api)
            count +=1
    return count


def getROIsFrom3DMasks(section_data_set_ids,structure_id_list,reference_space_id,resolution,width,height,image_api,sync_api,tree,rsp):
    """ This function gets the ROIs of strucure ids from a list of section data sets.
    Parameters:
    -----------
    section_data_set_ids: list of int
        The list of section data set ids to get the Rois from.
    structure_id_list: list of int
        The list of trageted structure ids.
    reference_space_id: int
        The id of the reference space 9 is for coronal data, 10 is for sagittal data.
    resolution: int
        The resolution of the reference space that is being used (10, 25, 50, 100).
    width: int
        The width of the ROI.
    height: int
        The height of the ROI.
    Returns:
    --------
    count of the ROIS
    """
    #get structures from their id.
    structure_infos = tree.get_structures_by_id(structure_id_list)
    count = 0
    for st in structure_infos:
        # A complete 3D mask for one structure
        whole_cortex_mask = rsp.make_structure_mask([st['id']])
        ct = getROIsFrom3DMasks_per_structure(whole_cortex_mask,
                                            section_data_set_ids=section_data_set_ids,
                                            reference_space_id=reference_space_id,
                                            structure_acronym=st['acronym'],
                                            structure_id=st['id'],
                                            width=width,
                                            height=height,
                                            resolution = resolution,
                                            image_api = image_api,
                                            sync_api = sync_api,
                                            )
        count += ct
    return count

if __name__ == '__main__':

    ### Configs ###
    reference_space_key = 'annotation/ccf_2017' # the reference space key to use
    resolution = 25 # the resolution to use options are 10, 25, 50, 100
    rows_list = [] # the list of rows to store rois information vefore converting to dataframe
    # The structure data set ids to use. Chosen by hand because of the errors in that can occur in the samples. All of the data is Coronal.
    structure_data_set_ids_list = [70928385,71249069,71836787,71836878,72081516,72119628,73520964,73521804,73636030,73771240,75042244,75650864,76135829,77413698,79488931,79591637,79912554]
    # The structure data set ids to extract the rois from.
    structure_ids_list = [1056, #ANcr1
                        507, # MOB
                        726 # DG
                        ]
    reference_space_id = 9 # the reference space id to use. 9 is for coronal data, 10 is for sagittal data.
    width = 334 # the width of the ROI.
    height = 334 # the height of the ROI.
    ### End of configs ###

    ### Arguments ###
    # pars structure_data_set_ids_list,structure_ids_list,reference_space_id,resolution
    parser = argparse.ArgumentParser(description='This script gets the ROIs of strucure ids from a list of section data sets.')
    parser.add_argument('-s', '--structure_data_set_ids_list', nargs="*", type=int, help='list of structure data set ids')
    parser.add_argument('-i', '--structure_ids_list',nargs="*", type=int, help='list of structure ids')
    parser.add_argument('-r', '--reference_space_id', type=int, help='reference space id')
    parser.add_argument('-R', '--resolution', type=int, help='resolution')
    parser.add_argument('-w', '--width', type=int, help='width')
    parser.add_argument('-H', '--height', type=int, help='height')
    args = parser.parse_args()
    structure_data_set_ids_list = args.structure_data_set_ids_list if args.structure_data_set_ids_list != None else structure_data_set_ids_list
    structure_ids_list = args.structure_ids_list if args.structure_ids_list != None else structure_ids_list
    reference_space_id = int(args.reference_space_id) if args.reference_space_id != None else reference_space_id
    resolution = int(args.resolution) if args.resolution != None else resolution
    width = int(args.width) if args.width != None else width
    height = int(args.height) if args.height != None else height
    ### End of arguments ###


    


    ### APIs and instances ###
    image_api = ImageDownloadApi()
    sync_api = SynchronizationApi()
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=os.path.join("allen_ccf", "manifest.json")) 
    tree = rspc.get_structure_tree(structure_graph_id=1) # The structure tree to use for extracting information about the structures (acronym, names, ids ...)
    annotation, meta = rspc.get_annotation_volume() # the annotation volume to use for getting the labels of regions
    rsp = rspc.get_reference_space() 

    
    getROIsFrom3DMasks(structure_data_set_ids_list,structure_ids_list,reference_space_id,resolution,width,height,image_api,sync_api,tree,rsp) 
    #getROIsFrom3DMasks([74511839],[726,1056,507],9,25,334,334) 
   