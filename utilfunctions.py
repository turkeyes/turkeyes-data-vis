import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import ast
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import math

def get_heatmap_for_image(filename,D,base_path,sigma=50,toplot=False, resize = None):
    
    imfilename = os.path.join(base_path, filename)
    im = Image.open(imfilename).convert('RGB')

    if (not resize is None):
        [width0,height0] = im.size
        scaleF = max(width0/float(resize[0]),height0/float(resize[1]))
        resize = [int(width0/scaleF),int(height0/scaleF)]
        im = im.resize(resize)
        sigma = int(sigma/scaleF)
        #print(sigma)

    [width,height] = im.size
    
    temp = np.zeros([height,width])
    for i in range(len(D)):
        coords = D[i]['coords']
        if coords[1]<=height and coords[0]<=width:
            temp[coords[1],coords[0]] += 1
        
    res = scipy.ndimage.filters.gaussian_filter(temp,[sigma,sigma]);
    
    if toplot:
        plt.imshow(res);
    
    return res,im

def get_heatmap_for_image_multcoords(filename,D,base_path,sigma=50,toplot=False):
    
    imfilename = os.path.join(base_path, filename)
    #im = Image.open(imfilename)
    im = Image.open(imfilename).convert('RGB')
    [width,height] = im.size
    
    temp = np.zeros([height,width])
    for i in range(len(D)):
        coords = D[i]['coords'] # multiple coords
        if len(coords)>2: # to get list
            for ii in range(len(coords)):
                if int(coords[ii][1])<height and int(coords[ii][0])<width:
                    temp[int(coords[ii][1]),int(coords[ii][0])] += 1
        
    res = scipy.ndimage.filters.gaussian_filter(temp,[sigma,sigma]);
    
    if toplot:
        plt.imshow(res);
    
    return res,im

def get_heatmap_for_coords(filename,coords,base_path,sigma=50,toplot=False):
    
    imfilename = os.path.join(base_path, filename)
    im = Image.open(imfilename).convert('RGB')
    [width,height] = im.size
    
    temp = np.zeros([height,width])
    for ii in range(len(coords)):
        if type(coords[ii])==list:
            if int(coords[ii][1])<height and int(coords[ii][0])<width:
                temp[int(coords[ii][1]),int(coords[ii][0])] += 1
        
    res = scipy.ndimage.filters.gaussian_filter(temp,[sigma,sigma]);
    
    if toplot:
        plt.imshow(res);
    
    return res,im

def get_processed_heatmap_for_import(filename, imdir, impdir):
    
    impdirname = filename.split('.')[0]
    imfilename = os.path.join(imdir, filename)
    impfilename = os.path.join(impdir, filename)
    im = Image.open(imfilename).convert('RGB')
    impim = np.asarray(Image.open(impfilename).convert('RGB').resize(im.size))[:,:,0]
    
    return impim, im

def get_heatmap_for_import(filename, imdir, maskdir):
    
    check_img_type = lambda file: (lambda file: not file.startswith('.') or file.endswith('png') or file.endswith('jpg')
                               or file.endswith('jpeg'))(file.lower())
    get_heatmap_norm = lambda heatmap: (heatmap-np.min(heatmap))/float(np.max(heatmap)-np.min(heatmap))
    maskfolder = os.path.join(maskdir, filename)
    masknames = [file for file in os.listdir(maskfolder) if check_img_type(file)]
    imfilename = os.path.join(imdir, filename)
    im = Image.open(imfilename).convert('RGB')
    width, height = im.size
    temp = np.zeros([height,width], dtype=float)
    for maskname in masknames:
        maskfilename = os.path.join(maskfolder, maskname)
        mask = np.asarray(Image.open(maskfilename).resize(im.size))
        temp+=mask
    
    temp = get_heatmap_norm(temp)
    
    return temp, im

def get_processed_heatmap_for_zoom(filename, imdir, impdir):
    
    impdirname = filename.split('.')[0]
    imfilename = os.path.join(imdir, filename)
    impim = np.load(os.path.join(impdir, impdirname, 'all_avg.npy'))
    im = Image.open(imfilename).convert('RGB').resize(impim.shape[::-1])
    
    return impim, im

def cc(s_map,gt_map):
    M1 = np.divide(s_map - np.mean(s_map), np.std(s_map))
    M2 = np.divide(gt_map - np.mean(gt_map), np.std(gt_map))
    return np.corrcoef(M1.reshape(-1),M2.reshape(-1))[0][1]

def nss(s_map,gt_locs):
    # s_map is a distribution; gt_locs is a list of (x,y) coordinates 
    w,h = s_map.shape
    s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
    gt_locs_valid = [[int(coord[1]),int(coord[0])] for coord in gt_locs if coord[0]<h and coord[1]<w]
    nss_vals = [s_map_norm[coord[0],coord[1]] for coord in gt_locs_valid]
    return np.mean(nss_vals)

