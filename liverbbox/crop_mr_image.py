import numpy as np
from scipy.ndimage import label, labeled_comprehension, find_objects, binary_erosion, binary_opening
from scipy.signal  import argrelextrema, convolve

def crop_mr_image(mr_vol,              #3d numpy array in LPS orientation
                  voxsize_mr,
                  size_mm_max          = [None, None, None],
                  air_abd_ratio        = 0.2,
                  tissue_ratio         = 0.02,
                  liver_height_max     = 400,
                  zstart_margin_mm        = 0.):
   """
   crop the mr image so that the cropped image contains the whole abdomen in each slice. 
   The slices of the cropped image should start from the lung and end at the hip.
   
   Parameters 
   -----------
   
   mr_vol: 3d numpy array in LPS orientation which contains the image values.
   
   voxsize_mr: the MR voxel size. 
   
   size_mm_max: a list with three elements, the maximum size (mm) of the cropped image in three dimensions. (default: [None, None, None])
   
   tissue_ratio: the ratio used to threshould the soft tissues, air should be excluded by this threshold. (default: -150)
                 Threshold = ratio * maximum_value_in_MR_vol
   
   air_abd_ratio: the ratio used to threshold the y-direnction profile of the mr image so that we can find the bounds in y direnction which seperates abdomen from air
   
   liver_height_max: the maximum size of liver in z dimension. (default: 400)
   
   zstart_margin_mm: the margin left for the starting of the slices in z-direction. (default: 0)
   
   
   Return
   ------------
   
   The bounding box used to crop the input CT image
   
   """

   xsize_mm_max = size_mm_max[0]
   ysize_mm_max = size_mm_max[1]
   zsize_mm_max = size_mm_max[2]

   mr_size = mr_vol.shape
   #########################################################################################################################
   # crop the image based on the zstart and zend and then find the largest connected region (abdomen) in the cropped image #
   # try to remove as much as background and arms away                                                                     #
   #########################################################################################################################
   
   # find the bounding values in y direction
   # calcualte the difference of y profile for the MR image
   y_profile = np.sum(mr_vol, (0,2))
   y_profile = convolve(y_profile, np.ones(7), 'same')
   y_prof_diff = convolve(y_profile, np.concatenate((np.ones(30),np.zeros(1),-np.ones(30))), 'same')
   
   # find the global minima and local maxima of y profile
   y_prof_diff_argmin = np.argmin(y_prof_diff)
   y_prof_diff_arg_local_max = argrelextrema(y_prof_diff, np.greater, order = 70)[0]
   order_tmp = 70
   while len(y_prof_diff_arg_local_max) <= 1 and order_tmp != 2:
     order_tmp = max(order_tmp - 10, 2)
     y_prof_diff_arg_local_max = argrelextrema(y_prof_diff, np.greater, order = order_tmp)[0]
   
   # find the maximum value in the abdominal range of y_profile_diff
   if len(y_prof_diff_arg_local_max) == 1:
     y_abd_max = y_prof_diff_arg_local_max[0]
   else: 
     y_prof_diff_local_max_sort = np.sort(y_prof_diff[y_prof_diff_arg_local_max])
     possbile_y_abd_maxs = y_prof_diff_arg_local_max[np.where(y_prof_diff[y_prof_diff_arg_local_max] > 0.5*y_prof_diff_local_max_sort[-2])]
     possbile_y_abd_maxs = possbile_y_abd_maxs[np.where(possbile_y_abd_maxs < y_prof_diff_argmin-100)]

     diff_tmp = 100
     while len(possbile_y_abd_maxs) < 1:
       diff_tmp = diff_tmp - 10
       possbile_y_abd_maxs = y_prof_diff_arg_local_max[np.where(y_prof_diff[y_prof_diff_arg_local_max] > 0.5*y_prof_diff_local_max_sort[-2])]
       possbile_y_abd_maxs = possbile_y_abd_maxs[np.where(possbile_y_abd_maxs < y_prof_diff_argmin-diff_tmp)]
     
     possbile_y_abd_maxs = possbile_y_abd_maxs[np.where(possbile_y_abd_maxs < 0.4 * mr_size[1])]
     
     # use the global maxima if possbile_y_abd_maxs is null
     if len(possbile_y_abd_maxs) == 0:
       y_abd_max = np.argmax(y_prof_diff)
       if y_abd_max >= 0.4 * mr_size[1]:
         y_abd_max = 0
     else:
       if len(possbile_y_abd_maxs) > 1 and np.all(y_prof_diff[possbile_y_abd_maxs[-2]:possbile_y_abd_maxs[-1]+1] > 0.5*y_prof_diff[possbile_y_abd_maxs[-1]]):
         y_abd_max = possbile_y_abd_maxs[-2]
       elif len(possbile_y_abd_maxs) > 1 and y_prof_diff[possbile_y_abd_maxs[-1]] < 0.5 * y_prof_diff[possbile_y_abd_maxs[-2]]:
         y_abd_max = possbile_y_abd_maxs[-2]   
       else:
         y_abd_max = possbile_y_abd_maxs[-1]

   # find the starting slice index of the abdomial region in y direction
   possible_y_abd_starts = np.array(np.where(y_prof_diff < 0.45*y_prof_diff[y_abd_max]))
   possible_y_abd_starts = possible_y_abd_starts[np.where(possible_y_abd_starts < y_abd_max)]
   Th_tmp = 0.45
   while len(possible_y_abd_starts) == 0 and Th_tmp < 1.0:
     Th_tmp = Th_tmp + 0.05
     possible_y_abd_starts = np.array(np.where(y_prof_diff < Th_tmp*y_prof_diff[y_abd_max]))
     possible_y_abd_starts = possible_y_abd_starts[np.where(possible_y_abd_starts < y_abd_max)]
   # set the starting point in y direction to zero if possible starting points still can't be found
   if len(possible_y_abd_starts) == 0:
     y_abd_start = 0
   else:
     y_abd_start = possible_y_abd_starts[-1]

   # find the possible stopping slice index of the abdominal region in y direction 
   possbile_y_abd_ends = np.array(np.where(y_prof_diff > 0.6*y_prof_diff[y_prof_diff_argmin]))
   possbile_y_abd_ends = possbile_y_abd_ends[np.where(possbile_y_abd_ends > y_prof_diff_argmin)]
   ratio_tmp = 0.6
   while len(possbile_y_abd_ends) < 1:
     ratio_tmp = ratio_tmp + 0.05
     if ratio_tmp > 1.0:   
       break
     possbile_y_abd_ends = np.array(np.where(y_prof_diff > ratio_tmp*y_prof_diff[y_prof_diff_argmin]))
     possbile_y_abd_ends = possbile_y_abd_ends[np.where(possbile_y_abd_ends > y_prof_diff_argmin)]
   
   # find the stopping slice index of the abdominal region in y direction
   if ratio_tmp <= 1.0: 
     possbile_y_abd_ends_diff = convolve(possbile_y_abd_ends, np.concatenate((np.ones(1),-np.ones(1))), 'same')
     idx_div = np.where(possbile_y_abd_ends_diff > 1)[0]
     if len(idx_div) > 1:
       possbile_y_abd_ends = possbile_y_abd_ends[idx_div[-1]:]
     if len(possbile_y_abd_ends) == 0:
       y_abd_end = y_prof_diff_argmin
     else:
       y_abd_end = possbile_y_abd_ends[0]
   else:
     y_abd_end = mr_size[1]

   slice_abd_y = slice(y_abd_start, y_abd_end, None)

   
   # create final bounding box
   zstart = max(0, mr_vol.shape[2] - int(liver_height_max/voxsize_mr[2]) - int(zstart_margin_mm/voxsize_mr[2]))
   zend   = mr_vol.shape[2]  
   slice_z = slice(zstart, zend, None)
   
   # crop image
   mr_vol_cropped = mr_vol[:,:,slice_z]
   
   # crop around the soft tissue in LP direction 
   bin_mr_vol = (mr_vol_cropped > tissue_ratio * np.max(mr_vol_cropped))
   
   #binary opening image
   bin_mr_vol = binary_erosion(bin_mr_vol, np.ones((5,5,5)))
   bin_mr_vol = bin_mr_vol.astype('int16')
   
   # find biggest connected regions
   labeled_array, nlbl = label(bin_mr_vol)

   # calculated volumes of connected region 
   labels = np.arange(1, nlbl+1)

   nvox   = labeled_comprehension(mr_vol_cropped, labeled_array, labels, len, int, -1)

   # bounding box of biggest connected region in soft tissue binary image
   bbox = find_objects(labeled_array == labels[np.argmax(nvox)])[0]

   tmp0 = slice(bbox[0].start, bbox[0].stop, None)
   tmp1 = slice(bbox[1].start, bbox[1].stop, None)
   
   slice_y = slice(max(tmp1.start, slice_abd_y.start), min(tmp1.stop, slice_abd_y.stop), None)
   
   bbox_final = (tmp0,slice_y,slice_z)
   
   #######################################################################################
   #ensure the cropped image size does not exceed the maximum size in mm
   #######################################################################################
   
   #if the xsize_mm of the final cropped image is larger than xsize_mm_max, make sure xsize_mm = xsize_mm_max
   if xsize_mm_max is not None:
     xsize = bbox_final[0].stop - bbox_final[0].start
     if xsize * voxsize_mr[0] > xsize_mm_max: 
       offset_x_mm = xsize * voxsize_mr[0] - xsize_mm_max
       delta_x_start = np.ceil(offset_x_mm / (2 * voxsize_mr[0])).astype('int16')
       delta_x_stop = np.ceil(offset_x_mm / (2 * voxsize_mr[0])).astype('int16')
       bbox0_check = slice(bbox_final[0].start + delta_x_start, bbox_final[0].stop - delta_x_stop, None)
     else:
       bbox0_check = bbox_final[0]
   else:
     bbox0_check = bbox_final[0]
   
   #if the ysize_mm of the final cropped image is larger than ysize_mm_max, make sure ysize_mm = ysize_mm_max   
   if ysize_mm_max is not None:
     ysize = bbox_final[1].stop - bbox_final[1].start
     if ysize * voxsize_mr[1] > ysize_mm_max: 
       offset_y_mm = ysize * voxsize_mr[1] - ysize_mm_max
       delta_y_start = np.ceil(offset_y_mm / (2 * voxsize_mr[1])).astype('int16')
       delta_y_stop = np.ceil(offset_y_mm / (2 * voxsize_mr[1])).astype('int16')
       bbox1_check = slice(bbox_final[1].start + delta_y_start, bbox_final[1].stop - delta_y_stop, None)  
     else:
       bbox1_check = bbox_final[1]       
   else:
     bbox1_check = bbox_final[1]
 
   #if the zsize_mm of the final cropped image is larger than zsize_mm_max, crop the image so that zsize_mm = zsize_mm_max  
   if zsize_mm_max is not None:
     zsize = bbox_final[2].stop - bbox_final[2].start
     if zsize * voxsize_mr[2] > zsize_mm_max:
       delta_z_start = np.ceil((zsize * voxsize_mr[2] - zsize_mm_max) / voxsize_mr[2]).astype('int16')
       bbox2_check = slice(bbox_final[2].start + delta_z_start, bbox_final[2].stop, None)
     else:
       bbox2_check = bbox_final[2]
   else:
     bbox2_check = bbox_final[2]
     
   bbox_final_check = (bbox0_check, bbox1_check, bbox2_check)

   return bbox_final_check