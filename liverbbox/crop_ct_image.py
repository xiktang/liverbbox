import numpy as np
from scipy.ndimage import label, labeled_comprehension, find_objects, binary_erosion, binary_opening
from scipy.signal  import argrelextrema, convolve

def crop_ct_image(ct_vol,              #3d numpy array in LPS orientation
                  voxsize_ct,
                  size_mm_max          = [None, None, None],
                  tissue_th            = -150,
                  lung_th              = -600,
                  bone_th              = 500,
                  bone_hip_ratio       = 0.7, 
                  liver_height_max     = 400,
                  liver_dome_margin_mm = 70.,
                  hip_margin_mm        = 0.):
   """
   crop the ct image so that the cropped image contains the whole abdomen in each slice. 
   The slices of the cropped image should start from the lung and end at the hip.
   
   Parameters 
   -----------
   
   ct_vol: 3d numpy array in LPS orientation which contains the image values.
   
   voxsize_ct: the CT voxel size. 
   
   size_mm_max: a list with three elements, the maximum size (mm) of the cropped image in three dimensions. (default: [None, None, None])
   
   tissue_th: the threshold for the soft tissues and bones, air should be excluded by this threshold. (default: -150)
   
   lung_th: the threshould for the air inside the lung and from the background, soft tissues and bones should be excluded through this threshold. (default: -600)
   
   bone_th: the threshould for bones, soft tissues and air should be excluded by this threshold. (default: 500)
   
   bone_hip_ratio: the ratio used to set the threshold for the values which may belong to hip. (default: 0.7)
   
   liver_height_max: the maximum size of liver in z direction. (default: 400)
   
   liver_dome_margin_mm: the margin left for the liver dome. (default: 70)
   
   hip_margin_mm: the margin left for the hip. (default: 0)
   
   
   Return
   ------------
   
   The bounding box used to crop the input CT image
   
   """

   xsize_mm_max = size_mm_max[0]
   ysize_mm_max = size_mm_max[1]
   zsize_mm_max = size_mm_max[2]
   
   ###############################################################################################################
   #binarize the image and find the largest connected region, which is the abodomen. 
   #Crop the image in x and y direction according to the bounding box generated from the largest connected region
   ###############################################################################################################

   # binarize image 
   bin_ct_vol = (ct_vol > tissue_th)
 
   # erode bin_ct_vol 
   bin_ct_vol = binary_erosion(bin_ct_vol)

   # find biggest connected regions
   labeled_array, nlbl = label(bin_ct_vol)

   # calculated volumes of connected region 
   labels = np.arange(1, nlbl+1)
   nvox   = labeled_comprehension(ct_vol, labeled_array, labels, len, int, -1)

   # bounding box of biggest connected region in soft tissue binary image
   bbox1 = find_objects(labeled_array == labels[np.argmax(nvox)])[0]

   ct_vol2 = np.zeros(ct_vol.shape) + ct_vol.min()
   ct_vol2[bbox1] = ct_vol[bbox1]

   ###############################################################################
   # try to find lungs as symmetrical blobs with low intensity
   # get binary "air image" - only for calibrating HUs!
   ###############################################################################
   
   # create a new image in which the top slice is set to soft tissue HU
   # in case the FOV of the image cuts the lungs, the air in the lung is connected to background
   ct_vol3 = ct_vol2.copy()
   ct_vol3[:,:,-1] = 0

   # binarize the image 
   bin_air_ct_vol = (ct_vol3 < lung_th)
   
   # perform binary opening to avoid that lung is connected to background via nasal cavities
   bin_air_ct_vol_tmp = bin_air_ct_vol[:,:,:-1]
   bin_air_ct_vol_tmp = binary_opening(bin_air_ct_vol_tmp, np.ones((5,5,5)))
   bin_air_ct_vol[:,:,:-1] = bin_air_ct_vol_tmp	 
	 
   # pad the binary air mask to ensure that the background is not separated into several parts
   bin_air_ct_vol_pad = np.pad(bin_air_ct_vol, ((3,3),(3,3),(0,0)), mode='constant', constant_values  = 1)
   
   # label the binary air image 
   labeled_air_array_pad, nlbl_air = label(bin_air_ct_vol_pad)
   labeled_air_array = labeled_air_array_pad[3:-3, 3:-3, :]
   labels_air = np.arange(1, nlbl_air+1)
   
   #calculate the number of voxels in each labeled region
   nvox_air = labeled_comprehension(ct_vol3, labeled_air_array, labels_air, len, int, -1)
   air_volumes = nvox_air * np.prod(voxsize_ct)

   #find the air mask excluding the air from the background, then sum the air mask along the x and y axises to obtain the air profile.
   #The air profile value from the main part of lung should be much higher than from other parts. The approximate liver dome slice can be found by 
   #finding the threshold of the air profile.
   air_volumes_tmp = air_volumes.copy()
   air_volumes_tmp.sort()
   air_mask_abd = np.logical_and(labeled_air_array != labels_air[air_volumes == air_volumes_tmp[-1]][0], bin_air_ct_vol == 1)
   air_abd_profile = convolve(air_mask_abd.sum(0).sum(0),np.ones(7), 'same')
   air_profile_max = np.max(air_abd_profile)
   air_profile_min = np.min(air_abd_profile)   
   Th = air_profile_min + 0.5*(air_profile_max - air_profile_min)
   range_lung = np.where(air_abd_profile > Th)[0]
   range_lung_diff = convolve(range_lung,np.array([1,-1]), 'same')
   idx_range_starts = np.where(range_lung_diff != 1)[0]
   if len(idx_range_starts) > 1:
     if np.any(air_abd_profile[range_lung[idx_range_starts[1]-1]:range_lung[idx_range_starts[1]]] < air_profile_min + 0.2*(air_profile_max - air_profile_min)):
       liver_dome_sl_1 = range_lung[idx_range_starts[1]]
     else:
       liver_dome_sl_1 = range_lung[0]
   else:
     liver_dome_sl_1 = range_lung[0]
   
   # find the x and y coordinates of the central point in the abdomen air mask
   bbox_air_abd = find_objects(air_mask_abd)[0]
   cent_air_abd_x = (bbox_air_abd[0].start + bbox_air_abd[0].stop + 1)//2
   cent_air_abd_y = (bbox_air_abd[1].start + bbox_air_abd[1].stop + 1)//2 
   
   # find the lung with the largest lower bound between two lungs, which should be lower than the liver dome slice
   possbile_lung_labels = np.array([labels_air[air_volumes == air_volumes_tmp[-2]][0], labels_air[air_volumes == air_volumes_tmp[-3]][0]])
   bbox_lung_0 = find_objects(labeled_air_array == possbile_lung_labels[0])[0]
   bbox_lung_1 = find_objects(labeled_air_array == possbile_lung_labels[1])[0]
   lower_bounds_z = [bbox_lung_0[2].start, bbox_lung_1[2].start]
   lung_label = possbile_lung_labels[np.argmax(lower_bounds_z)]
   bbox_lung  = find_objects(labeled_air_array == lung_label)[0]

   liver_dome_sl_2 = bbox_lung[2].start
   
   #take the maximum value between the two candidates of the approximate liver dome slices
   liver_dome_sl = max(liver_dome_sl_1, liver_dome_sl_2)

   ##########################################################################################
   # look at bone image to find hip
   ##########################################################################################
   
   #binarize the image to obtain the bone mask
   bin_bone_ct_vol = (ct_vol > bone_th)
   bin_bone_ct_vol = bin_bone_ct_vol.astype('int16')
	 
   #sum the bone mask along x and y axises to obtain a 1d bone profile along z axis
   #smooth the bone profile through convolution to remove noise 
   bone_profile = convolve(bin_bone_ct_vol.sum(0).sum(0),np.ones(7), 'same')[:liver_dome_sl]
   
   #find the maximum the slice where the bone profile reaches its maximum value
   slice_bone_max = np.argmax(bone_profile)
   
   #find the slice numbers of the local minimum values among the bone profile
   slice_bone_local_min = argrelextrema(bone_profile, np.less, order = 15)[0]
   
   #in case the order is too large and no local minima is found, decrease the order until at least one local minima is found
   order_tmp = 15
   while len(slice_bone_local_min) == 0:
     order_tmp = order_tmp - 3
     order_tmp = max(1, order_tmp)
     slice_bone_local_min = argrelextrema(bone_profile, np.less, order = order_tmp)[0]
   
   #find the slice with the smallest value among all local minima 
   bone_local_min = bone_profile[slice_bone_local_min]
   idx_sort_min = np.argsort(bone_local_min)

   slice_bone_min = slice_bone_local_min[idx_sort_min[0]]
     
   #find the first and second largest local maxima. If there is only one local maxima, take this maxima as the largest local maxima.
   slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = 15)[0]
   order_tmp = 15
   while len(slice_bone_local_max) == 0:
     order_tmp = order_tmp - 3
     order_tmp = max(1, order_tmp)
     slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = order_tmp)[0]
   bone_local_max = bone_profile[slice_bone_local_max]
   idx_sort = np.argsort(bone_local_max)
   if len(idx_sort) < 2:
     slice_second_local_max = slice_bone_local_max[idx_sort[-1]]
   else:
     slice_second_local_max = slice_bone_local_max[idx_sort[-2]]
     
   #if the slice with the minimum value is smaller than the slices with the maximum value and with the second largest local maxima,
   #truncate the bone profile from the slice with the second largest local maxima and find the slice with the minimum value in the truncated 
   #bone profile. 
   if slice_bone_min < slice_bone_max:
     if slice_bone_min < slice_second_local_max:
       bone_profile_tmp = bone_profile[slice_second_local_max:]
       slice_bone_min = slice_second_local_max + np.argmin(bone_profile_tmp)
       
   #find the slices with the local maximum values in the bone profile
   slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = 5)[0]
   order_tmp = 5
   while len(slice_bone_local_max) == 0:
     order_tmp = order_tmp - 1
     order_tmp = max(1, order_tmp) 
     slice_bone_local_max = argrelextrema(bone_profile, np.greater, order = order_tmp)[0]     
   
   #The possible hip slices should have the bone profile values over 'bone_hip_factor' times the minimum bone profile value
   #The possbile hip slices should also belong to the slices with the local maximum values in the bone profile
   #possible_hip_slices = np.where(bone_profile > bone_hip_factor*bone_profile[slice_bone_min])[0]     
   possible_hip_slices = np.where(bone_profile > bone_hip_ratio*(bone_profile[slice_bone_max]-bone_profile[slice_bone_min]) + bone_profile[slice_bone_min])[0] 
   possible_slice_bone_local_max = np.intersect1d(slice_bone_local_max, possible_hip_slices)
   bone_hip_ratio_tmp = bone_hip_ratio
   while len(possible_hip_slices) == 0 or len(possible_slice_bone_local_max) == 0:
     bone_hip_ratio_tmp = bone_hip_ratio_tmp - 0.1
     bone_hip_ratio_tmp = max(bone_hip_ratio_tmp, 0.05)
     possible_hip_slices = np.where(bone_profile > bone_hip_ratio_tmp*(bone_profile[slice_bone_max]-bone_profile[slice_bone_min]) + bone_profile[slice_bone_min])[0] 
     possible_slice_bone_local_max = np.intersect1d(slice_bone_local_max, possible_hip_slices)

   #The hip slice should be closet to the slice with the minimum bone profile value among the possible hip slices which belong to local maximas
   #in the bone profile
   if len(np.where(possible_slice_bone_local_max < slice_bone_min)[0]) < 2:
     hip_slice = 0
   else:
     hip_slice = possible_slice_bone_local_max[np.where(possible_slice_bone_local_max < slice_bone_min)][-1]
   
   #the distance between the slice of liver dome and hip slice should not be over the defined maximum liver height
   if (liver_dome_sl - hip_slice) * voxsize_ct[2] > liver_height_max:
     hip_slice = liver_dome_sl - int(liver_height_max/voxsize_ct[2])

   #########################################################################################################################
   # crop the image based on the zstart and zend and then find the largest connected region (abdomen) in the cropped image #
   # try to remove as much as background and arms away                                                                     #
   #########################################################################################################################
   
   # create final bounding box
   zstart = max(0, hip_slice - int(hip_margin_mm/voxsize_ct[2]))
   zend   = min(ct_vol.shape[2], liver_dome_sl + int(liver_dome_margin_mm/voxsize_ct[2]))

   bbox = (bbox1[0],bbox1[1],slice(zstart,zend,None))

   # crop image
   ct_vol_cropped = ct_vol[bbox]

   # do another crop around the soft tissue in LP direction 
   bin_ct_vol = (ct_vol_cropped > tissue_th)

   #binary opening image
   bin_ct_vol = binary_erosion(bin_ct_vol, np.ones((5,5,5)))
   bin_ct_vol = bin_ct_vol.astype('int16')
   
   # find biggest connected regions
   labeled_array, nlbl = label(bin_ct_vol)

   # calculated volumes of connected region 
   labels = np.arange(1, nlbl+1)
   nvox   = labeled_comprehension(ct_vol_cropped, labeled_array, labels, len, int, -1)

   # bounding box of biggest connected region in soft tissue binary image
   bbox2 = find_objects(labeled_array == labels[np.argmax(nvox)])[0]

   tmp0 = slice(bbox[0].start + bbox2[0].start, bbox[0].start + bbox2[0].stop, None)
   tmp1 = slice(bbox[1].start + bbox2[1].start, bbox[1].start + bbox2[1].stop, None)

   bbox_final = (tmp0,tmp1,bbox[2])
   
   #######################################################################################
   #ensure the cropped image size does not exceed the maximum size in mm
   #######################################################################################
   
   #if the xsize_mm of the final cropped image is larger than xsize_mm_max, make sure xsize_mm = xsize_mm_max
   if xsize_mm_max is not None:
     xsize = bbox_final[0].stop - bbox_final[0].start
     if xsize * voxsize_ct[0] > xsize_mm_max: 
       offset_x_mm = xsize * voxsize_ct[0] - xsize_mm_max
       delta_x_start = np.ceil((cent_air_abd_x - bbox_final[0].start) / xsize * offset_x_mm / voxsize_ct[0]).astype('int16')
       delta_x_stop = np.ceil((bbox_final[0].stop - cent_air_abd_x) / xsize * offset_x_mm / voxsize_ct[0]).astype('int16')  
       bbox0_check = slice(bbox_final[0].start + delta_x_start, bbox_final[0].stop - delta_x_stop, None)
     else:
       bbox0_check = bbox_final[0]
   else:
     bbox0_check = bbox_final[0]
   
   #if the ysize_mm of the final cropped image is larger than ysize_mm_max, make sure ysize_mm = ysize_mm_max   
   if ysize_mm_max is not None:
     ysize = bbox_final[1].stop - bbox_final[1].start
     if ysize * voxsize_ct[1] > ysize_mm_max: 
       offset_y_mm = ysize * voxsize_ct[1] - ysize_mm_max
       delta_y_start = np.ceil((cent_air_abd_y - bbox_final[1].start) / ysize * offset_y_mm / voxsize_ct[1]).astype('int16')
       delta_y_stop = np.ceil((bbox_final[1].stop - cent_air_abd_y) / ysize * offset_y_mm / voxsize_ct[1]).astype('int16')   
       bbox1_check = slice(bbox_final[1].start + delta_y_start, bbox_final[1].stop - delta_y_stop, None)  
     else:
       bbox1_check = bbox_final[1]       
   else:
     bbox1_check = bbox_final[1]
 
   #if the zsize_mm of the final cropped image is larger than zsize_mm_max, crop the image so that zsize_mm = zsize_mm_max  
   if zsize_mm_max is not None:
     zsize = bbox_final[2].stop - bbox_final[2].start
     if zsize * voxsize_ct[2] > zsize_mm_max:
       delta_z_start = np.ceil((zsize * voxsize_ct[2] - zsize_mm_max) / voxsize_ct[2]).astype('int16')
       bbox2_check = slice(bbox_final[2].start + delta_z_start, bbox_final[2].stop, None)
     else:
       bbox2_check = bbox_final[2]
   else:
     bbox2_check = bbox_final[2]
     
   bbox_final_check = (bbox0_check, bbox1_check, bbox2_check)

   return bbox_final_check