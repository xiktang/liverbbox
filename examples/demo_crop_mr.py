import numpy as np
from liverbbox import crop_ct_image, crop_mr_image

parser = ArgumentParser()
parser.add_argument('img_np_path',   help = 'the path of the image saved in numpy array')
parser.add_argument('voxsize_img',    help = 'the voxel size of the image')
parser.add_argument('size_mm_max', default = [489, 408, 408], help = 'the maximum size (mm) of the cropped image in three dimensions')
parser.add_argument('tissue_ratio', default =  0.01, help = 'the ratio used to threshould the soft tissues, air should be excluded by this threshold')
parser.add_argument('liver_height_max', default =  400, help = 'the maximum size of liver in z dimension')
parser.add_argument('zstart_margin_mm', default =  30., help = 'the margin left for the starting of the slices in z-direction')

img_np_path = args.img_np_path
voxsize_img = args.voxsize_img
size_mm_max = args.size_mm_max
tissue_ratio = args.tissue_ratio
liver_height_max = args.liver_height_max
zstart_margin_mm = args.zstart_margin_mm

img_vol = np.load(img_np_path)
bbox = crop_mr_image(img_vol,  #3d numpy array in LPS orientation
                   voxsize_img,
                   size_mm_max      = size_mm_max,
                   tissue_ratio     = tissue_ratio,
                   liver_height_max = liver_height_max,
                   zstart_margin_mm = zstart_margin_mm)  
img_crop = img_vol[bbox]