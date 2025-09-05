# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:38:30 2024

@author: hkleikamp
"""


#%% Modules

import pySPM

import os
from pathlib import Path
import subprocess


import math
import numpy as np
import pandas as pd


from scipy.optimize import curve_fit
from scipy.signal import find_peaks,find_peaks_cwt,  savgol_filter
from sklearn.preprocessing import robust_scale
from scipy.sparse import csr_matrix 


from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

#plotting 
import textalloc as ta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'


# %% change directory to script directory (should work on windows and mac)
from inspect import getsourcefile
basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())


#%% Parameters

#Filepaths

itmfiles=[
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A1.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A2.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_A1.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_A2(images).itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_C1(images).itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot I_B1.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot II_A1.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot II_B1.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_A2(images).itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_B1.itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_C1(images).itm",
"F:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_C1(sputter).itm"]



grd_exe="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe" #path to Grd executable 





#%% Command line
import sys
if not hasattr(sys,'ps1'): #checks if code is executed from command line
    
    import argparse

    parser = argparse.ArgumentParser(
                        prog='SIMSpy',
                        description='TOF-SIMS data analysis from .grd, see: https://github.com/hbckleikamp/SIMSpy')
    
    
    #Filepaths
    parser.add_argument("-i", "--itmfile", required = True,  help="Required: IonTOF .itm file output")
    parser.add_argument("--grd_exe", required = False,  default="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe",
                        help="if no .grd files are present, they will be made with this executable. if not available, request access to IonTOF")

    

    args = parser.parse_args()
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(args)
    
itmfiles=list(itmfiles)

for itmfile in itmfiles:
    I = pySPM.ITM(itmfile)
    

    
    #construct grd if missing
    grdfile=itmfile+".grd"
    if not os.path.exists(grdfile):
        command='"'+grd_exe+'"'+' "'+itmfile+'"'
        print(command)
        stdout, stderr =subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


    

