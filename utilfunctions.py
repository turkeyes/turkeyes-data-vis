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

def get_heatmap_for_image(filename,D,base_path,sigma=50,toplot=False):
    
    imfilename = os.path.join(base_path, filename)
    im = Image.open(imfilename).convert('RGB')
    [width,height] = im.size
    
    temp = np.zeros([height,width])
    for i in range(len(D)):
        coords = D[i]['coords']
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

