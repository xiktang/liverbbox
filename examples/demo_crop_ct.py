import numpy as np
from liverbbox import crop_ct_image, crop_mr_image

parser = ArgumentParser()
parser.add_argument('img_np_path',   help = 'the path of the image saved in numpy array')
parser.add_argument('voxsize_img',    help = 'the voxel size of the image')
parser.add_argument('size_mm_max', default = [489, 408, 408], help = 'the maximum size (mm) of the cropped image in three dimensions')
parser.add_argument('tissue_th', default =  -150, help = 'the threshold for the soft tissues and bones, air should be excluded by this threshold')
parser.add_argument('lung_th', default =  -600, help = 'the threshould for the air inside the lung and from the background, soft tissues and bones should be excluded through this threshold')
parser.add_argument('bone_th', default =  500, help = 'the threshould for bones, soft tissues and air should be excluded by this threshold')
parser.add_argument('bone_hip_ratio', default =  0.7, help = 'the ratio used to set the threshold for the values which may belong to hip')
parser.add_argument('liver_height_max', default =  400, help = 'the maximum size of liver in z direction')
parser.add_argument('liver_dome_margin_mm', default =  400, help = 'the margin left for the liver dome')
parser.add_argument('hip_margin_mm', default =  400, help = 'the margin left for the hip')

img_np_path = args.img_np_path
voxsize_img = args.voxsize_img
size_mm_max = args.size_mm_max
tissue_th = args.tissue_th
lung_th = args.lung_th
bone_th = args.bone_th
bone_hip_ratio = args.bone_hip_ratio
liver_height_max = args.liver_height_max
liver_dome_margin_mm = args.liver_dome_margin_mm
hip_margin_mm = args.hip_margin_mm

img_vol = np.load(img_np_path)
bbox = crop_ct_image(img_vol,  #3d numpy array in LPS orientation
                     voxsize_img,
                     size_mm_max = size_mm_max,
                     tissue_th  = tissue_th,
                     lung_th   = lung_th,
                     bone_th   = bone_th,
                     bone_hip_ratio = bone_hip_ratio, 
                     liver_height_max = liver_height_max,
                     liver_dome_margin_mm = liver_dome_margin_mm,
                     hip_margin_mm        = hip_margin_mm)
img_crop = img_vol[bbox]