# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:57:20 2025

@author: e_kle
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:36:18 2023

@author: hkleikamp
"""
### Modules ###
from pathlib import Path
import os
from inspect import getsourcefile

# change directory to script directory (should work on windows and mac)
os.chdir(str(Path(os.path.abspath(getsourcefile(lambda:0))).parents[0]))
sippy_dir=os.getcwd()
basedir=os.getcwd()


import pySPM
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


#%%



ITA_files=[] #put ITA files here




for file_ix,file in enumerate(ITA_files):


    fs=Path(file).stem
    A = pySPM.ITA(file)
    

    ## Plot snapshot
    meta_S=A.root.goto('Meta/Video Snapshot').dict_list()
    meta_I=A.root.goto('Meta/SI Image').dict_list()
    
    sx,  sy  =meta_S['res_x']['ulong']              ,meta_S['res_y']['ulong']               #pixels
    x_um,y_um=meta_S["fieldofview_x"]["float"]*10**6,meta_S["fieldofview_y"]["float"]*10**6 #real FOV
    zoom_factor=meta_I["zoomfactor"]["float"]
    scale_x, scale_y = x_um/sx, y_um/sy  #pixels to len
    xlen=A.size["real"]["x"]*1e6/zoom_factor/scale_x #in pixels
    ylen=A.size["real"]["y"]*1e6/zoom_factor/scale_y #in pixels
    

    try:
        img = np.array(A.root.goto('Meta/Video Snapshot/imagedata').get_data()).reshape((sy, sx, -1)) #error in module
     
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) 
        ax.imshow(img)
        plt.title(fs)
    
        
        
    
        plt.xticks(ax.get_xticks(),(ax.get_xticks()*scale_x).astype(int))
        plt.yticks(ax.get_yticks(),(ax.get_yticks()*scale_y).astype(int))
        plt.xlim(0,sx)
        plt.ylim(0,sy)
    
        #add scan patch
        
        xdiff=(meta_S['stageposition_x']["float"]-meta_I['stageposition_x']["float"])*1e6/scale_x
        ydiff=(meta_S['stageposition_y']["float"]-meta_I['stageposition_y']["float"])*1e6/scale_y
        xstart=(sx-xlen+xdiff)/2
        ystart=(sy-ylen+ydiff)/2
    

        plt.scatter(sx/2,sy/2,s=5,c="red",marker="x")
        
        rect = patches.Rectangle((xstart,ystart), xlen, ylen, linewidth=1, edgecolor='black', facecolor='none',linestyle="--")
        ax.add_patch(rect)
        
        plt.xlabel(u"\u03bcm")
        plt.ylabel(u"\u03bcm")
        plt.legend(["center","SI-zone"],loc=[1.02,0.45])
     
        fig.savefig(file.replace(Path(file).suffix,"_snapshot.png"),dpi=1000)

        
    except:
        pass
    
    ##### plot SI
    X, Y = A.size['pixels']['x'], A.size['pixels']['y']
    arr=np.array(A.root.goto('Meta/SI Image/intensdata').get_data("f"), dtype=np.float32).reshape((Y, X))


  

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    i=ax.imshow(arr,cmap="Greys_r")
    plt.colorbar(i)
    plt.title(fs)

    plt.xticks(ax.get_xticks(),(ax.get_xticks()/ax.get_xticks().max()*xlen*scale_x).astype(int))
    plt.yticks(ax.get_yticks(),(ax.get_yticks()/ax.get_yticks().max()*ylen*scale_y).astype(int)[::-1])
    plt.xlim(0,X)
    plt.ylim(Y,0)

    plt.xlabel(u"\u03bcm")
    plt.ylabel(u"\u03bcm")

    fig.savefig(file.replace(Path(file).suffix,"_SI.png"),dpi=1000)
    

   