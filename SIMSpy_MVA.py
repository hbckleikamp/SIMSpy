# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 10:13:11 2025

@author: e_kle
"""

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
import json

import math
import numpy as np
from numpy.linalg import norm
import pandas as pd



from skmisc import loess #specific module for weighted Loess

from scipy.optimize import curve_fit, nnls
from scipy.signal import find_peaks,find_peaks_cwt, savgol_filter
from scipy.sparse import csr_matrix, coo_matrix
from scipy.ndimage import gaussian_filter

from sklearn.preprocessing import robust_scale, maxabs_scale, StandardScaler #,MaxAbsScaler
from sklearn.decomposition import PCA,TruncatedSVD,NMF
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.neighbors import KDTree


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
base_vars=list(locals().copy().keys()) #base variables

#%% Parameters

#Filepaths
itmfiles=""

itmfiles=["E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_C1(sputter).itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A2.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_A1.itm",
#"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_A2(images).itm",
#"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_C1(images).itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot I_B1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot II_A1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot II_B1.itm",
#"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_A2(images).itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_B1.itm"]


#itmfiles=["E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A1.itm"]

grd_exe="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe" #path to Grd executable 
Output_folder=""
write_params=True
load_params=""

#initial dimension redution
max_mass=1000            #truncate mass
min_scans,max_scans=0,0 #truncate scans 
min_x,max_x=0,0
min_y,max_y=0,0
min_width_ratio,max_width_ratio=0,0       #remove based on ratio to expected peakwidth
bin_pixels=0
bin_scans=0 
bin_tof=0
Remove_bin_edges=True   #when corners do not satifsy complete bins, trim them

#peak picking
prominence=10
distance=20       #minimum distance between peaks
extend_window=20  #extend around peak bases
cwt_w=10          #wavelet transform window
Peak_deconvolution=True

#Calibration
ppm_cal=100         #maximum deviation for finding internal calibrants
Substrate="InOSnSi"         #list of elements that are present on substrate
Substrate_Calibrants=str(Path(basedir,"Substrate_Calibrants.csv"))  #list of ions coming from substrate
Calibrants=str(Path(basedir,"Calibrants.csv"))                      #list of typical background ions


#ROI
ROI_peaks=1000
ROI_clusters=0
ROI_dimensions=3 
ROI_bin_pixels=2                #integer or fraction 
ROI_bin_scans=2 
ROI_scaling="Poisson" #Jaccard           # Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"

#depth profile
normalize=True
smoothing=3

#isotope detection
Detect_isotopes=True
isotope_range=[-4,6] 
ippm=100          #ppm tolerance for detecting isotopes
co_normalize=False #normalize during cosine correlation of depth profiles
co_smoothing=3     #smooth during cosine correlation of depth profiles
min_cosine=0.9

#MVA dimension reduction
data_reduction="binning" #"binning" #or peak_picking

#if binning
MVA_bin_tof=5           # bins tof (mass)
MVA_bin_pixels=2 #3       # bins in 2 directions: 2->, 3->9 data reduction
MVA_bin_scans=2   #5       # bin scans
min_count=2         # remove bins with les than x counts

#MVA
MVA_peaks=0 #10000 #maxmimum nr of peaks used for MVA
MVA_dimensions=[2] #[1,2,3]  #can be list
MVA_components=5          # number of components
MVA_methods=["NMF","PCA"]
MVA_scaling="Standard"    # Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"
reconstruct_intensity=True          # Convert components back to total intensity 

#Top_quantile=0.25 #False      # [0-1, float] only retain mass channels that have top X abundance quantile 
Remove_zeros=True       #
PCA_algorithm="arpack"  # 'arpack' or 'random'
Varimax=False           # perform Varimax rotation in PCA, improves scale of mass components
NMF_algorithm='nndsvd'

#%% store parameters
params={}
[params.update({k:v}) for k,v in locals().copy().items() if k not in base_vars and k[0]!="_" and k not in ["base_vars","params"]]

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
    parser.add_argument("-o", "--Output_folder", required = False, default="", help="Output folder")
    parser.add_argument("--write_params",  required = False, default=True, help="write parameters used to file")
    parser.add_argument("--load_params",  required = False, default="", help="use params from a file")
    
    #initial dimension redution
    parser.add_argument("--max_mass", required = False, default=1000, help="removes masses above this threshold")
    parser.add_argument("--max_scans", required = False, default=False, help="removes scans above this threshold")
    parser.add_argument("--min_width_ratio", required = False, default=0.1, help="remove masses below the ratio to expected peakwidth")
    parser.add_argument("--max_width_ratio", required = False, default=5, help="remove masses above the ratio to expected peakwidth")
    parser.add_argument("--Remove_bin_edges", required = False, default=True, help="if bins are not fully filled, ignore bin (this could remove last x/y pixels or last scan)")
    
    #calibration 
    parser.add_argument("--ppm_cal", required = False, default=500, help="maximum ppm devatation for selecting calibrants")
    parser.add_argument("--Substrate", required = False, default="", help="Add Substrate calibrants that contain these elements")
    parser.add_argument("--Substrate_Calibrants", required = False, default=str(Path(basedir,"Substrate_Calibrants.csv")) , help="List of Substrate Calibrants or path to file")
    parser.add_argument("--Calibrants", required = False, default=str(Path(basedir,"Calibrants.csv"))   , help="List of Sample Calibrants or path to file")
    
    #peak picking
    parser.add_argument("--prominence", required = False, default=10, help="minimum prominence for peak detection")
    parser.add_argument("--distance", required = False, default=20, help="minimum distance for peak detection")
    parser.add_argument("--extend_window", required = False, default=20, help="peak extension")
    parser.add_argument("--cwt_w", required = False, default=10, help="window for continuous wavelet transform peak filtering")

    #ROI detection
    parser.add_argument("--ROI_clusters", required = False, default=2, help="Regions of interest (ROI) detected")
    parser.add_argument("--ROI_peaks", required = False, default=1000, help="number of peaks considered for ROI detection")
    parser.add_argument("--ROI_dimensions", required = False, default=2, help="ROI dimensions (2 or 3")
    parser.add_argument("--ROI_scaling", required = False, default="Jaccard", help=' Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"')
    parser.add_argument("--ROI_bin_pixels", required = False, default=3, help='if binning: bin pixels in x & y direction')    
    parser.add_argument("--ROI_bin_scans", required = False, default=5, help='if binning: bin frames (scans)')   

    #depth profile
    parser.add_argument("--Peak_deconvolution", required = False, default=True, help=' Split mixed peaks with gaussian deconvolution')
    parser.add_argument("--normalize", required = False, default=False, help=' normalize depth profile to total count')
    parser.add_argument("--smoothing", required = False, default=0, help=' moving average smoothing window')

    #Isotope detection
    parser.add_argument("--Detect_isotopes", required = False, default=True, help=' Perform isotope detection')
    parser.add_argument("--isotope_range", required = False, default=[-2,6] , help=' minimum and maximum isotope considered')
    parser.add_argument("--ippm", required = False, default=True, help=' ppm mass error used for identifying isotopes')
    parser.add_argument("--min_cosine", required = False, default=0.9, help=' minimum cosine similarity of isotopes compared to monoisotope peak')
    parser.add_argument("--co_normalize", required = False, default=True, help=' normalize during cosine correlation of depth profiles')
    parser.add_argument("--co_smoothing", required = False, default=3, help=' smooth during cosine correlation of depth profiles')

    #MVA data reduction
    parser.add_argument("--data_reduction", required = False, default="binning", help='MVA data reduction: "binning" or "peak_picking"')    
    parser.add_argument("--MVA_bin_tof", required = False, default=5, help='if binning: bin tof channels')    
    parser.add_argument("--MVA_bin_pixels", required = False, default=3, help='if binning: bin pixels in x & y direction')    
    parser.add_argument("--MVA_bin_scans", required = False, default=5, help='if binning: bin frames (scans)')    
    parser.add_argument("--min_count", required = False, default=2, help='remove bins/peaks with fewer counts')    
    parser.add_argument("--MVA_peaks", required = False, default=False, help="maximum number of peaks considered fo MVA")
    
    #MVA parameters
    parser.add_argument("--MVA_dimensions", required = False, default=3, help="single or list of dimensions (1,2,3)")    
    parser.add_argument("--MVA_methods", required = False, default="NMF", help='single or list of methods (NMF, PCA)')    
    parser.add_argument("--n_components", required = False, default=5, help='number of components')    
    parser.add_argument("--MVA_scaling", required = False, default="MinMax", help='Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"')    
    
    #parser.add_argument("--Top_quantile", required = False, default=False, help='only retain top abundance quantile for MVA')        
    parser.add_argument("--Remove_zeros", required = False, default=True, help='remove pure zero rows and columns')    
    parser.add_argument("--PCA_algorithm", required = False, default="arpack", help='PCA algorithm')    
    parser.add_argument("--NMF_algorithm", required = False, default="nndsvd", help='NMF algorithm')    
    parser.add_argument("--Varimax", required = False, default=False, help='Perform Varimax on PCA loadings')    


    args = parser.parse_args()
    params = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(params)

#%% update arguments

#update arguments from parameter file
if len(load_params): 
    if os.path.exists(load_params):
        with open(load_params, 'r') as f:
            jdict=json.load(params, f)
            locals().update(jdict)


if type(itmfiles)==str: itmfiles=itmfiles.split(",")

if type(MVA_dimensions)==int or type(MVA_dimensions)==float: MVA_dimensions=[int(MVA_dimensions)]
if type(MVA_methods)==str: MVA_methods=[i.strip() for i in MVA_methods.split(",")]
MVA_dimensions.sort()

isotope_range=np.arange(isotope_range[0],isotope_range[1]+1)

#get element masses
ifile=str(Path(basedir,"natural_isotope_abundances.tsv"))
emass=0.000548579909 #mass electron
elements=pd.read_csv(ifile,sep="\t") #Nist isotopic abundances and monisiotopic masses
elements=elements[elements["Standard Isotope"]].set_index("symbol")["Relative Atomic Mass"]
elements=pd.concat([elements,pd.Series([-emass,emass],index=["+","-"])]) #add charges

if type(itmfiles)==str:
    if os.isdir(itmfiles):
        itmfiles=[str(Path(itmfiles,i)) for i in os.listdir(itmfiles) if i.endswith(".itm")]        
    else: itmfiles=itmfiles.split(",")


#%% Functions

def m2c(m,sf,k0):    return np.round(  ( sf*np.sqrt(m) + k0 )  ).astype(int)
def m2cf(m,sf,k0):   return  ( sf*np.sqrt(m) + k0 )   #float version
def c2m(c,sf,k0,bin_tof=1):    return ((c*bin_tof - k0) / (sf)) ** 2   

    #add bin_tof function to c2m!

def residual(p,x,y): return (y-c2m(x,*p))/y
def centroid(x, y):  return np.sum(x * y) / np.sum(y)
def Gauss(x, a,mu, sig): return a*(1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))
def gaussian(x, mu, sig): return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

#https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
def create_ranges(a):
    l = a[:,1] - a[:,0]
    clens = l.cumsum()
    ids = np.ones(clens[-1],dtype=int)
    ids[0] = a[0,0]
    ids[clens[:-1]] = a[1:,0] - a[:-1,1]+1
    return ids.cumsum()

#vectorized find nearest mass
#https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
def find_closest(A, target): #returns index of closest array of A within target
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def parse_form(form): #chemical formular parser
    e,c,comps="","",[]
    for i in form:
        if i.isupper(): #new entry   
            if e: 
                if not c: c="1"
                comps.append([e,c])
            e,c=i,""         
        elif i.islower(): e+=i
        elif i.isdigit(): c+=i
    if e: 
        if not c: c="1"
        comps.append([e,c])
    
    cdf=pd.DataFrame(comps,columns=["elements","counts"]).set_index("elements").T.astype(int)
    cdf["+"]=form.count("+")
    cdf["-"]=form.count("-")
    return cdf

def getMz(form): #this could be vectorized for speed up in the future
    cdf=parse_form(form)
    return (cdf.values*elements.loc[cdf.columns].values).sum() / cdf[["+","-"]].sum(axis=1)
    

#https://stackoverflow.com/questions/51570512/minmax-scale-sparse-matrix-excluding-zero-elements
def scale_sparse_matrix_rows(s, lowval=0, highval=1):
    d = s.data

    lens = s.getnnz(axis=1)
    idx = np.r_[0,lens[:-1].cumsum()]

    maxs = np.maximum.reduceat(d, idx)
    mins = np.minimum.reduceat(d, idx)

    minsr = np.repeat(mins, lens)
    maxsr = np.repeat(maxs, lens)

    D = highval - lowval
    scaled_01_vals = (d - minsr)/(maxsr - minsr)
    d[:] = scaled_01_vals*D + lowval
 
#%%
def CalibrateGlobal(channels,calibrants,sf,k0,
                    ppm_cal=ppm_cal,
                    min_mass=10,
                    plot=True,
                    weights="",
                    bin_tof=1,
                    filename_add=""):
        

    #try: 
    
    
        # #test
        # sf,k0=I.sf,I.k0
        # channels=centroids
        # ppm_cal=200
        # min_mass=10
        # plot=True
        # weights=sSpectrum[ps[:,0]].values
        # bin_tof=1
        # filename_add=""

    fcalibrants=calibrants[calibrants>min_mass]
    q=np.argsort(channels)
    channels=channels[q]
    if len(weights): weights=weights[q]
    mp=c2m(channels,sf,k0,bin_tof=bin_tof)
    
    tree = KDTree(mp.reshape(1,-1).T, leaf_size=200) 

    l=tree.query_radius(fcalibrants.reshape(1,-1).T,r=fcalibrants*ppm_cal/1e6)
    caldf=pd.DataFrame(fcalibrants,columns=["c"])
    caldf["ix"]=[list(i) for i in l]
    caldf=caldf.explode("ix").dropna()
    caldf["ix"]=caldf["ix"].astype(int)
    caldf["mass"]=mp[caldf.ix]
    caldf["channels"]=channels[caldf.ix]
    if len(weights): caldf["intensity"]=weights[caldf.ix]
    caldf["ppm"]=(caldf["mass"]-caldf["c"])/caldf["c"]*1e6
    caldf=caldf.sort_values(by="mass")
    x,y=caldf["channels"],caldf["c"]
    y_pred=c2m(x,sf,k0,bin_tof=bin_tof)

    
    
    ### base recalibration if fit is bad

    if len(weights): r2 = r2_score(y, y_pred, sample_weight=caldf["intensity"].values)
    else:            r2 = r2_score(y, y_pred)
    if r2<0.95: 
        if len(weights): [sf,k0], _ = curve_fit(c2m,x,y,p0=[sf,k0,bin_tof],sigma=1/caldf["intensity"].values, absolute_sigma=True)
        else:            [sf,k0], _ = curve_fit(c2m,x,y,p0=[sf,k0,bin_tof])
        
        caldf["mass"]=c2m(caldf["channels"],sf,k0,bin_tof=bin_tof)
        caldf["ppm"]=(caldf["mass"]-caldf["c"])/caldf["c"]*1e6

    

    ### weighed Loess with quantile denoising
    x,y=caldf["mass"],caldf["ppm"]
    if len(weights):l=loess.loess(x, y, weights=np.log2(caldf["intensity"].values),span=0.9)
    else:           l=loess.loess(x, y,span=0.9)
    l.fit()
    p = l.predict(x).values
    
    #quantile denoise worst fits and fit again
    pm=p-caldf["ppm"].values
    q1,q3=np.percentile(pm,25),np.percentile(pm,75) 
    q=(pm<q3+1.5*(q3-q1)) & (pm>q3-1.5*(q3-q1))

    xq,yq=x[q],y[q]
    if len(weights):l=loess.loess(xq, yq, weights=np.log2(caldf["intensity"].values[q]),span=0.9)
    else:           l=loess.loess(xq, yq,span=0.9)
    l.fit()

    if plot:
        
        pq = l.predict(xq).values    
        post_ppms=yq-pq
        
            
        if len(weights):
            mpre=(caldf["ppm"]*caldf["intensity"]).sum()/caldf["intensity"].sum()
            mpost=(post_ppms*caldf["intensity"]).sum()/caldf["intensity"].sum()
            ampre=str(round(abs(caldf["ppm"]*caldf["intensity"]).sum()/caldf["intensity"].sum(),1))
            ampost=str(round(abs(post_ppms*caldf["intensity"]).sum()/caldf["intensity"].sum(),1))
        else:
            mpre=caldf["ppm"].mean()
            mpost=np.mean(post_ppms)
            ampre=str(round(sum(abs(caldf["ppm"]))/len(caldf),1))
            ampost=str(round(sum(abs(post_ppms))/len(caldf),1))
            
        #plotting
        fig,ax=plt.subplots()
        plt.scatter(x,caldf["ppm"],c=[(0.5, 0, 0, 0.3)],label="pre calibration")
        plt.scatter(xq,post_ppms,   c=[(0, 0.5, 0, 0.3)],label="post calibration")
        plt.plot(xq,pq,linestyle="--",color="grey",label="Loess fit")
        plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("ppm mass error")
        plt.title("global_calibration")
        fig.savefig(fs+filename_add+"_glob_cal_scat.png",bbox_inches="tight",dpi=300)
        plt.close()
    
        fig,ax=plt.subplots()
        y1, _, _ =plt.hist(caldf["ppm"],color=(0.5, 0, 0, 0.3))
        y2, _, _ =plt.hist(post_ppms,color=(0, 0.5, 0, 0.3))
        plt.vlines(mpre,0,np.hstack([y1,y2]).max(),color=(0.5, 0, 0, 1),linestyle='dashed')
        plt.vlines(mpost,0,np.hstack([y1,y2]).max(),color=(0, 0.5, 0, 1),linestyle='dashed')
        plt.xlabel("ppm mass error")
        plt.ylabel("frequency")
        plt.legend(["pre: mean "+str(round(mpre,1))+ ", abs "+ampre,
                    "post: mean "+str(round(mpost,1))+ ", abs "+ampost],
                    loc=[1.01,0])
        plt.title("global_calibration")
        fig.savefig(fs+filename_add+"_glob_cal_hist.png",bbox_inches="tight",dpi=300)
        plt.close()
        

    # except:
       
    #     print("calibration failed!, increase calibration ppm?")
    #     return I.sf,I.k0,c2m(centroids,sf,k0,bin_tof=bin_tof),np.array([0]*len(centroids))
        
    pd.DataFrame([[sf,k0]],columns=["sf","k0"]).to_csv(fs+"calib.csv")

    return sf,k0,xq,pq

#%%
def smooth_spectrum(ds):
    
    
    
    #either input a spectrum or a tof-table
    if   isinstance(ds,pd.DataFrame): Spectrum=np.bincount(ds["tof"].values)
    elif isinstance(ds,np.ndarray):     Spectrum=ds 
    else: print("wrong input type")
    
       
    #base smoothing on signal periodicity
    p=find_peaks(Spectrum)[0]
    u,c=np.unique(np.diff(p),return_counts=True)
    c_smoothing_window=u[np.argmax(c)]
    sSpectrum=pd.Series(Spectrum).rolling(window=c_smoothing_window).mean().fillna(0) 

    return sSpectrum,c_smoothing_window



#%%

#this function returns a smoothed spectrum, and picks singular peaks for calibration and resolution fitting
def pick_peaks(ds,
               
               rel_height=0.8,
               height_filter=0.3,
               
               extend_window=extend_window,
               cwt_w=cwt_w,
               
               plot_selected=False,
               plot_non_selected=False):
    
    # ""+1
    #%% test
    # rel_height=0.5
    # height_filter=0.1
    
    # extend_window=0
    # cwt_w=10
    # plot_selected=True #False #True
    # plot_non_selected=False #True

    # rel_height=0.8
    # height_filter=0.3



    

    sSpectrum,c_smoothing_window=smooth_spectrum(ds)

    #subtract baseline
    
    p,pi=find_peaks(sSpectrum,prominence=prominence,distance=distance)
    pro,lb,rb=pi["prominences"],pi["left_bases"],pi["right_bases"]
    
    
    s=np.argsort(rb-lb)     #sort by width?
    s=np.argsort(pro)[::-1] #or sort by prominence?
    p,pro,lb,rb=p[s],pro[s],lb[s],rb[s]


    good,bad=[],[]
    for ix,_ in enumerate(p):


        c=sSpectrum[lb[ix]:rb[ix]]
        x,y=c.index,c.values
        my=np.argmax(y)
        d=(y-y.max()*(1-rel_height))<0
        
        if (not d[0]) | (not d[-1]) :
            continue
        
        
        lx=np.argwhere(d[:my]).max()-extend_window
        rx=my+np.argwhere(d[my:]).min()+extend_window
        if lx<0: lx=0
        if rx>=len(d): rx=len(d)-1
        
        w=rx-lx
        lw,rw=x[lx],x[rx]
        r=c[lx:rx].values
  
        cwp=find_peaks_cwt(r,cwt_w) #flat window
        if len(cwp)>1: #more than one cwt peak
            

            
            cwph=y[cwp+lx]
            rh=cwph/cwph.max()
            
            if sum(rh>height_filter)-1: #cwt peak is above relative height of main peak
 
                
                #smoothed peak detection for picking noisey peaks
                scwp=find_peaks_cwt(savgol_filter(r,window_length=int(w/4),polyorder=2),widths=int(w/10))
                
                if len(scwp)>1:        #cwt detection with savgol smoothing
                    scwph=y[scwp+lx]
                    srh=scwph/cwph.max()
                    if sum(srh>height_filter)-1: #cwt peak is above relative height of main peak
                        #-> peak is not kept
                
                        if plot_non_selected: #testing
                            fig,ax=plt.subplots()
                            plt.plot(c.index,c.values)
                            plt.vlines([lw,rw],0,y.max(),color="grey")
                            plt.scatter(p[ix],y[my],color="red")
                            plt.scatter(c.index[cwp],c.values[cwp],color="orange") #plot cwt peaks
                            plt.legend(["counts","borders","peak","cwt peaks"])
                            plt.xlim(lw-200,rw+200)
                            plt.title("not selected" + str(ix))
                            
                            
                        bad.append([p[ix],lw,rw])
                       
                        continue

        if plot_selected: #testing
            fig,ax=plt.subplots()
            plt.plot(c.index,c.values)
            plt.vlines([lw,rw],0,y.max(),color="grey")
            plt.scatter(p[ix],y[my],color="red")
            plt.xlim(lw-200,rw+200)
            plt.title("selected" + str(ix))
                
        good.append([p[ix],lw,rw])
#%%

    return sSpectrum,np.array(good)-c_smoothing_window,np.array(bad)-c_smoothing_window



    
    
def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)
    
#%%

#testing function
def plot_around(f,sSpectrum,
                window=0.2 #in Da
                ):
    
        
    m=getMz(f).values
    l,u=m-window,m+window
    mass_points=(c2m(np.arange(len(sSpectrum)),sf,k0))
    q=(mass_points>l) & (mass_points<u)
    
    x,y=mass_points[q],sSpectrum[q]
    
    fig,ax=plt.subplots()
    plt.plot(x,y)
    ax.vlines(m,0,y.max(),color="r")
    plt.title(f)
    
    
#The number of bits needed to represent an integer n is given by rounding down log2(n) and then adding 1
def bits(x,neg=False):
    bitsize=np.array([8,16,32,64])
    dtypes=[np.uint8,np.uint16,np.uint32,np.uint64]
    if neg: dtypes=[np.int8,np.int16,np.int32,np.int64]
    return dtypes[np.argwhere(bitsize-(np.log2(x)+1)>0).min()]


def gaussian2(x, amplitude, mean, stddev):
    """Single Gaussian function"""
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def multi_gaussian(x, *params):
    """Sum of multiple Gaussian functions"""
    result = np.zeros_like(x).astype(float)
    # Each Gaussian has 3 parameters: amplitude, mean, stddev
    for i in range(0, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        stddev = params[i + 2]
        result += gaussian2(x, amplitude, mean, stddev).astype(float)
    return result


def deconvolve(pborders,sSpectrum,ext=10,gau_filt=20,cwt=5,r2=0.95,Plot=False,bin_tof=1,max_width_ratio=10):
    

    
    #needs
    #peak borders, columns: left_border, right_border
    #summed spectrum 
    #fitted peak resolution (xres, yres)

    print("Peak deconvolution")

    deconv_peaks=[]
    if isinstance(sSpectrum, pd.Series) or isinstance(sSpectrum, pd.DataFrame): sSpectrum=sSpectrum.values 
    
    for ix,b in enumerate(pborders):

        #print(ix)
        w=np.interp(b[1],xres/bin_tof,pres/bin_tof)
        sigma=w/2.355
        v=sSpectrum[b[0]-ext:b[1]+ext+1]
        if gau_filt: v=gaussian_filter(v,w/gau_filt) 
        cwp=find_peaks_cwt(v,w/cwt) 

        width_ratio=(b[1]-b[0])/w
        if width_ratio>max_width_ratio: continue


        #limit max peaks
        max_peaks=int(np.ceil((b[1]-b[0])/w*1.5))
        if len(cwp)>max_peaks: cwp=cwp[np.argsort(v[cwp])[::-1][:max_peaks]]
    
 #%%
        if len(cwp)<=1: 
            cwp=[np.argmax(v)]
            deconv_peaks.append(np.hstack([b[0]+cwp,
                                            v[cwp]]))
        else: 
            
   
                #initial fit 
                xs=np.arange(len(v))
                X=np.vstack([gaussian(xs,c,sigma) for c in cwp])
                coeffs,_= nnls(X.T, v)
                xt=(X.T*coeffs).T
                
                
                if Plot: #test plot
                    fig,ax=plt.subplots()
                    plt.plot(v)
              
                    ax.vlines(cwp,0,v.max(),color="grey")
                    for y in xt :
                        plt.plot(xs,y)
                    ax.set_xticklabels(c2m((b[0]+ax.get_xticks())*bin_tof,sf,k0).round(2))
                    plt.title(ix)
            

                #secondary fit
                lower_bounds= np.tile([0             ,0,sigma/2]   ,len(cwp))
                upper_bounds= np.tile([v.max()*2,len(v),sigma  ]   ,len(cwp))
                initial_guess=np.vstack([v[cwp],cwp,np.repeat(sigma,len(cwp))]).T.flatten() 
                
                
                try:
                
                    popt, pcov = curve_fit(multi_gaussian, np.arange(len(v)), v, 
                                          p0=initial_guess, 
                                          bounds=(lower_bounds, upper_bounds),
                                          maxfev=2000)
                    
                    
                    
                    #calc r2
                    y_pred=multi_gaussian(np.arange(len(v)),*popt)
                    rpopt=popt.reshape(len(cwp),-1)
       
                    r2v = r2_score(v, y_pred)
                    if r2v>r2: deconv_peaks.append(np.vstack([b[0]+rpopt[:,1],rpopt[:,0]]).T) 
                    else:       deconv_peaks.append(np.vstack([b[0]+np.argmax(v),v.max()]).T) 
                
                except:
                    print("Couldnt fit!")
                    deconv_peaks.append(np.vstack([b[0]+np.argmax(v),v.max()]).T) 
                
                

    
    return deconv_peaks
        

#%%

#parse calibrants
if type(Substrate_Calibrants)==str:
    if os.path.exists(Substrate_Calibrants): Substrate_Calibrants=pd.read_csv(Substrate_Calibrants).values.flatten().tolist()
    else: Substrate_Calibrants=Substrate_Calibrants.split(",")
    
if type(Calibrants)==str:
    if os.path.exists(Calibrants): Calibrants=pd.read_csv(Calibrants).values.flatten().tolist()
    else: Calibrants=Calibrants.split(",")

if type(Substrate)==str: Substrate=Substrate.split(",")
Substrate_allowed=pd.concat([parse_form(i) for i in Substrate]).columns 
Substrate_f=[i for i in Substrate_Calibrants if parse_form(i).columns.isin(Substrate_allowed).all()]
Calibrants+=Substrate_f

#%%

t=[]

for itmfile in itmfiles:
    
    I = pySPM.ITM(itmfile)
    
    ### Get ITM metadata ###

    #copy metadata to clipboard
    #I.Series(I.getValues()).explode().str.replace("\x00"," ").reset_index().astype(str).to_clipboard()
    
    xpix,ypix= math.ceil(I.size["pixels"]["x"]), math.ceil(I.size["pixels"]["y"])
    zoom_factor=float(I.get_value("Registration.Raster.Zoom.Factor")["string"])
    x_um,y_um=I.size["real"]["x"]*1e6/zoom_factor,I.size["real"]["y"]*1e6/zoom_factor
    
    scans=math.ceil(I.Nscan)
    sputtertime=I.get_value("Measurement.SputterTime")["float"] 
    old_xpix,old_ypix,old_scans=xpix,ypix,scans #store
    
    k0,sf=I.k0,I.sf
    mode_sign={"positive":"+","negative":"-"}.get((I.polarity).lower()) 
    calibrants=np.sort(np.array([getMz(i) for i in Calibrants if i.endswith(mode_sign)])) #shortended calibrants list

    Output_folder=str(Path(basedir,Path(itmfile).stem))
    fs=str(Path(Output_folder,Path(itmfile).stem))+"("+mode_sign+")"  #base filename for outputs (with polarity)
    if not os.path.exists(Output_folder): os.makedirs(Output_folder)
    
    if write_params: 
        with open(fs+".params", 'w') as f:
            json.dump(params, f)
    
    
    #construct grd if missing
    grdfile=itmfile+".grd"
    if not os.path.exists(grdfile):
        command='"'+grd_exe+'"'+' "'+itmfile+'"'
        print(command)
        stdout, stderr =subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    
    ### Unpack raw scans ###
    with open(grdfile,"rb") as fin:
        ds=pd.DataFrame(np.fromfile(fin,dtype=np.uint32).reshape(-1,5)[:,[0,2,3,4]],
                        columns=["scan","x","y","tof"]).astype(np.uint32)
    

 
    fig,ax=plt.subplots()
    plt.plot(ds.groupby("x").size())    
    plt.plot(ds.groupby("y").size())
    plt.ylabel("total counts")
    plt.xlabel("unbinned pixels")
    plt.legend(["x","y"])    
    plt.title("pixel counts")
    fig.savefig(fs+"_xy_counts.png")
    
    fig,ax=plt.subplots()
    plt.plot(ds.groupby("scan").size())
    plt.ylabel("total counts")
    plt.xlabel("unbinned scans")
    plt.title("scan counts")
    fig.savefig(fs+"_scan_counts.png")
    
    #truncate scans/pixels
    if max_scans>0: 
        ds=ds[ds["scan"]<=max_scans] 
        scans=max_scans
    if max_x>0: 
        ds=ds[ds["x"]<=max_x] 
        xpix=max_x
    if max_y>0: 
        ds=ds[ds["y"]<=max_y] 
        ypix=max_y
    
    #truncate scans/pixels and re-zero
    if min_scans>0: 
        ds=ds[ds["scan"]>=min_scans]
        ds["scan"]-=min_scans
        scans-=min_scans
    if min_x>0: 
        ds=ds[ds["x"]>=min_x]
        ds["x"]-=min_x
        xpix-=min_x
    if min_y>0: 
        ds=ds[ds["y"]>=min_y]
        ds["y"]-=min_y
        ypix-=min_y
            
    x_um=x_um*xpix/old_xpix
    y_um=y_um*ypix/old_ypix        
    sputtertime=sputtertime*scans/old_scans
    
    #trim bin edges
    if Remove_bin_edges:
        if bin_pixels: xpix-=xpix%bin_pixels
        if bin_pixels: ypix-=ypix%bin_pixels
        if bin_scans: scans-=scans%bin_scans
        ds= ds[ (ds.x<xpix)  & (ds.y<ypix)  & (ds.scan<scans)   ]
    
    
    



    ###### Global calibration ######
    
    sSpectrum,ps,pns=pick_peaks(ds)                                                                      #this only picks "single peaks"
    peaks=np.vstack([ps,pns])
    lb,rb=peaks[:,1],peaks[:,2]
    ds=ds[np.in1d(ds.tof,create_ranges(np.vstack([lb,rb]).T))] 
    centroids=np.array([centroid(sSpectrum[i[1]:i[2]].index,sSpectrum[i[1]:i[2]].values) for i in ps])   #calculate peak centroids
    qfin=np.isfinite(centroids)
    centroids=centroids[qfin]
    
    print("Calibrating")
    sf,k0,xcal,ycal=CalibrateGlobal(centroids,calibrants,sf,k0,ppm_cal=ppm_cal,weights=sSpectrum[ps[qfin,0]].values) #calibrate
    print("Calibration Done")
    #truncate mass
    if max_mass: ds=ds[c2m(ds["tof"],sf,k0)<max_mass] 
    np.save(fs+"_summed_spectrum",sSpectrum) #save summed spectrum
    
    ######  Get mass resolution ###### 
    
    
    #%% gussian filter of good hits here
    
    if min_width_ratio or max_width_ratio or Peak_deconvolution:

        print("Fitting peak resolution")


        gaufits=[]
        for p in ps:
            c=sSpectrum[p[1]:p[2]]
            x,y=np.arange(len(c)),c.values
            x0=np.argmax(y)

            popt, _ = curve_fit(Gauss, np.arange(len(y)),y,p0=[sSpectrum[p[0]],x0,(p[2]-p[1])/2.355])
            yGau=Gauss(x,*popt)
            r2=r2_score(y,yGau)
            if r2>0.9: gaufits.append([p[0],popt[2]*2.355])
                
        gaufits=np.vstack(gaufits)
        gaufits=gaufits[np.argsort(gaufits[:,0])]
        xres,yres=gaufits[:,0],gaufits[:,1]

         
        try:
            print("Gaussian resolution fit")
            lres=loess.loess(xres, yres,span=0.75)
            lres.fit()
            pres = lres.predict(xres).values
            r2=r2_score(yres,pres)

        
            
        except:
            print("Failed!")
            print("Linear resolution fit")
            #linear fit of peak resolution
            xres,yres=ps[:,0],(ps[:,2]-ps[:,1])#/2
            s=np.argsort(xres)
            xres,yres=xres[s],yres[s]    
            A = np.vstack([xres, np.ones(len(xres))]).T
            [a, b], r = np.linalg.lstsq(A, yres)[:2]
            
            xres=len(sSpectrum)
            pres=len(sSpectrum)*a+b
            r2 = 1 - r / (yres.size * yres.var())
    
    
        #plot resolution fit
        fig,ax=plt.subplots()
        plt.scatter(xres,yres)
        plt.plot(xres,pres,color="red",linestyle="--")
        plt.xlabel("mass channel")
        plt.ylabel("fwhm in channels")
        plt.legend(["single peaks","loessfit, r2: "+str(np.round(r2,3))])
        fig.savefig(fs+"_channel_res.png",bbox_inches="tight",dpi=300)

        plt.close()    
    
           
        ###### Filter channels on peakwidth ###### 
           
  
        pw=peaks[:,2]-peaks[:,1]
        rs=pw/np.interp(peaks[:,0],xres,pres)
        q=[True]*len(rs)
        if max_width_ratio: q=(q) & (rs<=max_width_ratio)
        if min_width_ratio: q=(q) & (rs<=min_width_ratio)
        lb,rb=peaks[q,1].astype(int),peaks[q,2].astype(int)
        
        fig,ax=plt.subplots()
        plt.hist(rs,bins=50,label="peak width ratio")
        if max_width_ratio: plt.vlines(min_width_ratio,0,ax.get_yticks()[-1],color="r",linestyle="--",label="min width ratio")
        if min_width_ratio: plt.vlines(max_width_ratio,0,ax.get_yticks()[-1],color="r",linestyle=":",label="max width ratio")
            
        plt.title("peak width distribution")
        plt.xlabel("ratio to expeted peakwidth")
        plt.legend()
        fig.savefig(fs+"_peak_widths.png",bbox_inches="tight",dpi=300)
        plt.close()
        
        ds=ds[np.in1d(ds.tof,create_ranges(np.vstack([lb,rb]).T))]
    #%%

    
    ###### Assign peaks ######
    
    pmat=np.zeros(peaks.max()+1,dtype=np.int64)
    pmat[create_ranges(peaks[:,1:])]=np.repeat(np.arange(len(peaks)),peaks[:,2]-peaks[:,1])
    ds["peak_bin"]=peaks[pmat[ds["tof"]],0]

   
    #peak deconvolution?
    

    #%%
       
    


    
    #%%
    
    
    
    ###### ROI detection #####
    
    
    
    if ROI_dimensions<=1 or ROI_clusters<=1:
        ds["ROI"]=0
    else:
    

        if Remove_bin_edges:
            xpix-=xpix%ROI_bin_pixels
            ypix-=ypix%ROI_bin_pixels
            if (bool(ROI_bin_scans)) & (ROI_dimensions==3): 
                    scans-=scans%ROI_bin_scans

            ds= ds[ (ds.x<xpix)  & (ds.y<ypix)  & (ds.scan<scans)   ]

    
        if ROI_dimensions==2: gcols=["x","y"]
        if ROI_dimensions==3: gcols=["x","y","scan"]
        col="peak_bin"
        
        #binning
        rds=ds.copy()
        if ROI_bin_pixels: rds[["x","y"]]=(rds[["x","y"]]/ROI_bin_pixels).astype(int)
        if ROI_bin_scans:  rds["scan"]=(rds["scan"]/ROI_bin_scans).astype(int) 
        
        cd=rds.groupby(gcols+[col]).size().to_frame("count").reset_index() 
        



                 
        if not Remove_bin_edges:
            
            if ROI_bin_pixels:
                cd["count"]/=(np.hstack([[ROI_bin_pixels]*int(xpix/ROI_bin_pixels),(xpix%ROI_bin_pixels)+1])[cd["x"]]/ROI_bin_pixels)
                cd["count"]/=(np.hstack([[ROI_bin_pixels]*int(ypix/ROI_bin_pixels),(ypix%ROI_bin_pixels)+1])[cd["y"]]/ROI_bin_pixels)
                if (bool(ROI_bin_scans)) & (ROI_dimensions==3):
                    cd["count"]/=(np.hstack([[ROI_bin_scans]*int(scans/ROI_bin_scans),(scans%ROI_bin_scans)+1])[cd["scan"]]/ROI_bin_scans)

        
        #limit to x top peaks
        cds=cd.groupby("peak_bin")["count"].sum()
        if ROI_peaks:
            if len(cds)>ROI_peaks: cd=cd[cd["peak_bin"].isin(cds.sort_values().index[:ROI_peaks])] 
        
        #construct peak matrix
        ut=np.unique(cd[col])
        uz=np.zeros(cd[col].max()+1,dtype=np.uint32)
        uz[ut]=np.arange(len(ut))
        xys=cd[gcols].values
        split_at=np.argwhere(np.any(np.diff(xys,axis=0)!=0,axis=1))[:,0]
        sxys=np.vstack([xys[split_at],xys[-1]])
        sx=np.array_split(cd[[col,"count"]].values.astype(np.uint32),split_at+1)
        szm=np.zeros([len(sxys),len(ut)],dtype=bits(cd["count"].max()))  #convert back to int
        for ix,i in enumerate(sx): szm[ix,uz[i[:,0]]]=i[:,1]
        
        #future: implement coo matrix here
        
        szm=csr_matrix(szm)
        #not spare first
        if ROI_scaling=="Standard":  szm=StandardScaler(with_mean=False).fit_transform(szm)
        if ROI_scaling=="Robust":    szm=robust_scale(szm,axis=0,with_centering=False)
        if ROI_scaling=="Poisson":  szm=szm/np.sqrt(szm.mean(axis=0))  #2D poisson scaling X/ sqrt mean  /np.sqrt(szm.mean(axis=1))
        if ROI_scaling=="Jaccard":  szm=szm.astype(bool).astype(int)
        szm=szm.T
        if ROI_scaling=="MinMax":    scale_sparse_matrix_rows(szm, lowval=0, highval=1) #szm=maxabs_scale(szm,axis=0)  #szm=MaxAbsScaler().fit_transform(szm) #doesnt work

        
        #Kmeans
        kmeans = KMeans(n_clusters=ROI_clusters, random_state=0, n_init="auto").fit(szm.T)
        rois=kmeans.labels_
        
        #map to space for indexing
        if ROI_dimensions==2: 
            zroi=np.zeros([cd.x.max()+1,cd.y.max()+1],dtype=np.int8)
            zroi[sxys[:,0],sxys[:,1]]=rois
        if ROI_dimensions==3: 
            zroi=np.zeros([cd.x.max()+1,cd.y.max()+1,cd.scan.max()+1],dtype=np.int8)
            zroi[sxys[:,0],sxys[:,1],sxys[:,2]]=rois

        np.save(fs+"_ROImap",zroi) #save ROI map
        
        #plot 2D ROIs
        if ROI_dimensions==2:
            fig,ax=plt.subplots()
            g=sns.heatmap(rois.reshape(math.ceil(xpix/ROI_bin_pixels),math.ceil(ypix/ROI_bin_pixels)).T,robust=True,cmap='RdBu_r')
            
            #update to pixels to micrometers
            ax.set_yticklabels((np.linspace(0,y_um,len(g.get_yticks()))).astype(int),rotation=0)
            ax.set_ylabel(r'y $\mu$m') 
            
            ax.set_xticklabels((np.linspace(0,x_um,len(g.get_xticks()))).astype(int),rotation=90)
            ax.set_xlabel(r'x $\mu$m')
            
            fig.savefig(fs+"_ROI_2D.png",dpi=300,bbox_inches="tight") #save 2D ROI plot
            plt.close()
   
            
            
            
    
        #plot 3D ROIs
        if ROI_dimensions==3:
            rdf=pd.DataFrame(np.hstack([sxys,rois.reshape(-1,1)]),columns=["x","y","z","r"])
            
            rdf["x"]=rdf["x"]*x_um/xpix*ROI_bin_pixels
            rdf["y"]=rdf["y"]*y_um/ypix*ROI_bin_pixels
            rdf["z"]=rdf["z"]*sputtertime/scans*ROI_bin_scans
            rdf["r"]=rdf["r"].astype(str)
            
         
            
            #plot all
            fig = px.scatter_3d(rdf, y='x', x='y', z='z',
                           color_discrete_sequence=px.colors.qualitative.Safe,        #Dark24/Vivid/Bold/Set1   
                          color='r', size_max=12,opacity=0.01) 
            
            fig.update_layout(scene = dict(xaxis=dict(title=dict(text='y [micrometer]')),
                                           yaxis=dict(title=dict(text='x [micrometer]')),
                                           zaxis=dict(title=dict(text='Sputter time [s]')),),
                                           margin=dict(r=20, b=10, l=10, t=10))
            
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.65, y=1.25, z=1.55)
            )
    
            ax_style = dict(showbackground =True,
                        backgroundcolor="white",
                        showgrid=False,
                        zeroline=False)
        
    
            fig.update_layout(template="none", width=600, height=600, font_size=11,
                              scene=dict(xaxis=ax_style, 
                                         yaxis=ax_style, 
                                         zaxis=ax_style,
                                         camera_eye=dict(x=1.85, y=1.85, z=1)))
                
            fig.update_scenes(zaxis_autorange="reversed")
            fig.update_layout(scene_camera=camera, title="ROI 3D combined segmentation")
            
            if rdf.z.max()>rdf.y.max() or rdf.x.max()>rdf.y.max():
                fig.update_scenes(aspectmode="cube")
            
            #fig.show()
            fig.write_html(fs+"_ROI_3D_combined.html") #save 3d ROI plot
        
            #plot separate segments
            roi_reorder=rdf["r"].drop_duplicates().values.astype(int)
            for ix_roi,roi in enumerate(roi_reorder):
                
                d=rdf[rdf["r"]==str(roi)]
                fig = go.Figure(data=[go.Scatter3d(x=d["y"], y=d["x"], z=d["z"],
                                       mode='markers',opacity=0.04)])
                
                
                fig.update_traces(marker=dict(line=dict(width=0),
                                  
                                            color=px.colors.qualitative.Safe[ix_roi])),
                      
    
                fig.update_layout(scene = dict(xaxis=dict(title=dict(text='y [micrometer]'),  range=[0,x_um]),
                                               yaxis=dict(title=dict(text='x [micrometer]'),  range=[0,y_um]),
                                               zaxis=dict(title=dict(text='Sputter time [s]'),range=[sputtertime,0]),),
                                               margin=dict(r=20, b=10, l=10, t=10))
                
                ax_style = dict(showbackground =True,
                            backgroundcolor="white",
                            showgrid=False,
                            zeroline=False)
            
            
            
                fig.update_layout(template="none", width=600, height=600, font_size=11,
                                  scene=dict(xaxis=ax_style, 
                                             yaxis=ax_style, 
                                             zaxis=ax_style,
                                             camera_eye=dict(x=1.85, y=1.85, z=1),
                                             aspectmode='cube',
                                             ))
        
                camera = dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.65, y=1.25, z=1.55)
                )
        
                if d.z.max()>d.y.max() or d.x.max()>d.y.max():
                    fig.update_scenes(aspectmode="cube")
        
                fig.update_layout(scene_camera=camera, title="ROI 3D segment "+str(roi))
                
                #fig.show()
                fig.write_html(fs+"_ROI_3D_segment"+str(roi)+".html") #save 3d ROI plot
    
   
      
    #%%
    ###### Depth profile ######
    
    
    #do depth profile per ROI
    if ROI_clusters:
        if ROI_dimensions==2: ds["ROI"]=zroi[(ds["x"]/ROI_bin_pixels).astype(int),
                                             (ds["y"]/ROI_bin_pixels).astype(int)]            #is this the right row column order?
        if ROI_dimensions==3: ds["ROI"]=zroi[(ds["x"]/ROI_bin_pixels).astype(int),
                                             (ds["y"]/ROI_bin_pixels).astype(int),
                                             (ds["scan"]/ROI_bin_scans).astype(int)] #is this the right row column order?
    else: ds["ROI"]=0
    cd=ds.groupby(["peak_bin","scan","ROI"]).size().to_frame("count").reset_index()
    
    
    biplot_roipeaks=[]
    for roi in range(ROI_clusters+1): #Depth profile per ROI
        
        rcd=cd[cd["ROI"]==roi]
        if not len(rcd): continue
        
        if normalize: #normalize to total count per scan
            rcd["count"]=rcd["count"]/rcd.groupby("scan")["count"].sum().loc[rcd.scan].values 
            
        if smoothing: #smooth with rolling mean
            scan_arr=np.arange(scans)
            missing_scans=[] #add missing scans
            for n,g in rcd.groupby("peak_bin",sort=False)["scan"]:
                if len(g)!=len(scan_arr):
                    ms=scan_arr[~np.in1d(scan_arr,g)]
                    missing_scans.append(np.vstack([np.repeat(n,len(ms)),ms]).T)
            if missing_scans:
                rcd=pd.concat([rcd,pd.DataFrame(np.vstack(missing_scans),columns=["peak_bin","scan"])]).fillna(0).astype(int).sort_values(by=["peak_bin","scan"]).reset_index(drop=True)
            rcd=rcd.groupby("peak_bin").rolling(smoothing).mean().reset_index()[['peak_bin', 'scan', 'count']].fillna(0) 
        
        rcd["peak_bin"]=c2m(rcd["peak_bin"],sf,k0)
        rcd["peak_bin"]=rcd["peak_bin"]*(1-np.interp(rcd["peak_bin"],xcal,ycal)/1e6)
                    
        #save depth ROI profile
        if ROI_clusters>1: rcd.to_csv(fs+"_ROI"+str(roi)+"_depth_profile.tsv",sep="\t")  
        else:              rcd.to_csv(fs+"_depth_profile.tsv",sep="\t")  
        

        biplot_roipeaks.append(rcd.groupby("peak_bin")["count"].sum())
        
        #save summed channels
        # summed_channels=ds[ds["ROI"]==roi].groupby(["tof","scan"]).size()
        # cdf=pd.DataFrame(np.vstack(np.unique(summed_channels.values,return_counts=True)).T,columns=["s","c"])
        
        rds=ds[ds["ROI"]==roi]
        ss,_=smooth_spectrum(rds)
        rps,pi=find_peaks(ss,distance,prominence=10)
        lb,rb=pi["left_bases"],pi["right_bases"]
        
        #remove overlapping lb rb
        pdf=pd.DataFrame(np.vstack([rps,lb,rb]).T,columns=["peak","lb","rb"])
        pdf["width"]=pdf["rb"]-pdf["lb"]
        pdf.sort_values(by=["lb","width"],ascending=[True,False])
        pdf=pdf.sort_values(by=["lb","width"],ascending=[True,False]).groupby("lb",sort=False).nth(0)
        rps,lb,rb=pdf.peak.values,pdf.lb.values,pdf.rb.values
   
            
        if Peak_deconvolution:    
            
            #merge overlapping peaks
            t=np.diff(rps)>np.interp(rps,xres,pres)[:-1]
            groups=np.hstack([0,np.cumsum(np.diff(np.hstack([-1,np.argwhere(t)[:,0],len(rps)-1])))])

            pborders=np.vstack([[lb[groups[i]],rb[groups[i+1]-1]] for i in np.arange(len(groups)-1)])

            deconv_peaks=deconvolve(pborders,ss)
            
                                 

            ROIpeaks=np.vstack(deconv_peaks)
        else: 
            ROIpeaks=ss[rps].reset_index().values
            
        ROIpeaks=pd.DataFrame(ROIpeaks,columns=["TOF","Apex"]) 
        ROIpeaks["FWHM"]=np.interp(ROIpeaks["TOF"],xres,pres)
        
        ROIpeaks["mass"]=c2m(ROIpeaks["TOF"],sf,k0)
        ROIpeaks["mass"]*=(1-np.interp(ROIpeaks["mass"],xcal,ycal)/1e6)

        

        
  
        #% Isotope detection
        if Detect_isotopes:
    
            imass=(ROIpeaks[["mass"]].values+(isotope_range*0.997).reshape(1,-1)).flatten()
            isos=np.tile(isotope_range,len(ROIpeaks))
            ixs=np.repeat(np.arange(len(ROIpeaks)),len(isotope_range))
            
            tree = KDTree(ROIpeaks["mass"].values.reshape(-1,1), leaf_size=200) 
            l=tree.query_radius(imass.reshape(-1,1), r=imass*ippm/1e6)
            links=np.vstack([np.vstack([np.repeat(ix,len(i)),i]).T for ix,i in enumerate(l) if len(i)])
    
            links=pd.DataFrame(links,columns=["ix","ri"])
            links["isotope"]=isos[links.ix]
            links["index"]=ixs[links.ix]
            links[["TOF","mass","Apex","FWHM"]]=ROIpeaks[["TOF","mass","Apex","FWHM"]].loc[links.ri,:].values
    
            #shady code?
            links=links.sort_values(by=["ix","index","Apex"],ascending=[True,True,False])
            links=links.groupby(["ix","index"],sort=False).nth(0)
            
            #%cosine similarity filtering
    
            tofs=create_ranges(np.vstack([links["TOF"]-links["FWHM"]/2,links["TOF"]+links["FWHM"]/2]).round(0).astype(int).T)
            rds=rds[rds.tof.isin(tofs)]
            rds=rds.groupby(["scan","tof"]).size().reset_index()
            rds.columns=["scan","tof","count"]
            
            #construct coo matrix
            utof,uscan=np.unique(rds.tof),np.unique(rds.scan)
            
            ztof=np.zeros(utof[-1]+1,dtype=np.uint64)
            ztof[utof]=np.arange(len(utof))
            rds["ztof"]=ztof[rds.tof]
            
            zscan=np.zeros(uscan[-1]+1,dtype=np.uint64)
            zscan[uscan]=np.arange(len(uscan))
            rds["zscan"]=zscan[rds.scan]
    
        
            rdscoo=coo_matrix((rds["count"], (rds["ztof"], rds["zscan"])), shape=(int(rds.ztof.max()+1), int(rds.scan.max()+1))).tocsr()
            
            if co_normalize:
                rss=np.asarray(rdscoo.sum(axis=0)).flatten()
                if co_smoothing: rss=np.convolve(rss, np.ones(smoothing), 'valid') / smoothing
                rss[rss==0]=1
            
            gs=[]
            for n,g in links.groupby("index"):
    
                l,r=g["TOF"]-g["FWHM"]/2,g["TOF"]+g["FWHM"]/2
                bs=np.vstack([l,r]).round(0).astype(int).T #clip
                cmat=[]
                for b in bs:
                    cols=ztof[np.clip(np.arange(*b),0,len(ztof)-1)]
                    dp=np.asarray(rdscoo[cols[cols>0],:].sum(axis=0)).flatten()
                    if co_smoothing: dp=np.convolve(dp, np.ones(smoothing), 'valid') / smoothing
                    if co_normalize: dp=dp/rss
                    cmat.append(dp)
                cmat=np.vstack(cmat)
                A=cmat[np.argwhere(g.isotope==0)[:,0],:]
                res = np.dot(A/norm(A, axis=1)[...,None],(cmat/norm(cmat,axis=1)[...,None]).T) #cosine sim
                g["cos"]=res.T
                gs.append(g)
                
    
            ROIpeaks=pd.concat(gs)
            ROIpeaks=ROIpeaks[ROIpeaks["cos"]>min_cosine]
            
        #write ROIpeaks

        if ROI_clusters>1: ROIpeaks.to_csv(fs+"_ROI"+str(roi)+"_peaks.tsv",sep="\t")  
        else:              ROIpeaks.to_csv(fs+"_peaks.tsv",sep="\t")  

    if ROI_clusters>1:
        ##### Depth profile Biplot ######
        
        #https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
        #https://statomics.github.io/HDDA/svd.html#7_SVD_and_Multi-Dimensional_Scaling_(MDS)   
        #https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html
        
        
        ss=pd.concat(biplot_roipeaks,axis=1).sort_index()
        ss.columns=[np.arange(ROI_clusters)]
        ss.index=np.round(ss.index,2)
        
        #principle component analysis
        ss=(ss-ss.min())/(ss.max()-ss.min()+1) #minmax scaling
        ss=ss.fillna(0)
        pca = PCA(n_components=2).fit(ss.T)
        X_reduced = pca.transform(ss.T)
        scores = X_reduced[:, :2]
        loadings=pca.components_.T
        pvars = pca.explained_variance_ratio_[:2] * 100
        
        # proportions of variance explained by axes
        k=10
        arrows = loadings * np.abs(scores).max(axis=0)
        st = (loadings ** 2).sum(axis=1).argsort()
        tops=st[-k:]
        arrows = loadings[tops]
        
        rdf=pd.DataFrame(np.hstack([sxys,rois.reshape(-1,1)]),columns=gcols+["r"])
        roi_reorder=rdf["r"].drop_duplicates().values.astype(int) 
        plt.figure(figsize=(5, 5))
        fig,ax=plt.subplots()
        for rx,i in enumerate(scores):
            c=tuple([int(i)/255 for i in px.colors.qualitative.Safe[roi_reorder[rx]][4:-1].split(", ")])
            plt.scatter(i[0],i[1],c=c,label=rx)
        ax.legend(loc=[1.05,0.02])
        plt.title("ROI biplot")
        
        # axis labels
        for i, axis in enumerate('xy'):
            getattr(plt, f'{axis}ticks')([])
            getattr(plt, f'{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')
        
        width = -0.001 * np.min([np.subtract(*plt.xlim()), np.subtract(*plt.ylim())])
        
        # features as arrows
        for a, arrow in enumerate(arrows):
            plt.arrow(0, 0, *arrow, color='k', alpha=0.5, width=width, ec='none',
                      length_includes_head=True)
        
        ta.allocate_text(fig,ax,arrows[:,0],arrows[:,1],ss.index[tops])
        
        plt.savefig(fs+"depth_profile_biplot.png",dpi=300,bbox_inches="tight")
        plt.close()
    
    ###### Dimension reduction ######
    

    
    if data_reduction=="peak_picking": 
        bin_tof,bin_pixels,bin_scans=1,1,1 #turn off binning
    
    #binning
    ds["tof"]=  (ds["tof"] /MVA_bin_tof   ).astype(int) 
    ds[["x","y"]]=(ds[["x","y"]]/MVA_bin_pixels).astype(int)
    ds["scan"]=  (ds["scan"] /MVA_bin_scans   ).astype(int) 
    ds=ds.astype(np.uint32)
    #xpix,ypix,scans= xpix/MVA_bin_pixels,ypix/MVA_bin_pixels,scans/MVA_bin_scans
    #%%

    ###### Create peak matrix #####
    for MVA_dimension in MVA_dimensions:
        
        
        if MVA_dimension==1: gcols=["scan"] #["scan","ROI"]
        if MVA_dimension==2: gcols=["x","y"]
        if MVA_dimension==3: gcols=["x","y","scan"]
        
        #turn off ROI for dimensions>1
        if MVA_dimension>1: 
            ROI_c=0
            ds["ROI"]=0 
        else:
            ROI_c=ROI_clusters
            
        if data_reduction=="peak_picking": col="peak_bin"
        if data_reduction=="binning":      col="tof"
        
        
        
        for roi in range(ROI_c+1):
            
            cd=ds[ds["ROI"]==roi].groupby(gcols+[col]).size().to_frame("count").reset_index()
            
            
            #limit to x top peaks or tof
            cds=cd.groupby(col)["count"].sum()
            if MVA_peaks: 
                if len(cds)>MVA_peaks: 
                    cd=cd[cd[col].isin(cds.sort_values().index[:MVA_peaks])] 
   
    
   
       
            if Remove_bin_edges:
                if (bool(MVA_bin_pixels)) & ("x" in cd.columns):   
                    if xpix%MVA_bin_pixels: cd=cd[cd.x!=cd.x.max()]
                if (bool(MVA_bin_pixels)) & ("y" in cd.columns):
                    if ypix%MVA_bin_pixels: cd=cd[cd.y!=cd.y.max()]
                if (bool(MVA_bin_scans)) & ("scan" in cd.columns):    
                    if scans%MVA_bin_scans:  cd=cd[cd.scan!=cd.scan.max()]
                     
            else: #correct for pixel edges & scan edges
                
                #correct for pixel edges & scan edges
                if (bool(MVA_bin_pixels)) & ("x" in cd.columns):      cd["count"]/=(np.hstack([[MVA_bin_pixels]*int(xpix/MVA_bin_pixels),xpix%MVA_bin_pixels+1])[cd["x"]]/MVA_bin_pixels)
                if (bool(MVA_bin_pixels)) & ("y" in cd.columns):      cd["count"]/=(np.hstack([[MVA_bin_pixels]*int(ypix/MVA_bin_pixels),ypix%MVA_bin_pixels+1])[cd["y"]]/MVA_bin_pixels)
                if (bool(MVA_bin_scans)) & ("scan" in cd.columns):    cd["count"]/=(np.hstack([[MVA_bin_scans]*int(scans/MVA_bin_scans),scans%MVA_bin_scans+1])[cd["scan"]]/MVA_bin_scans)
                              
    
       
        
       

            
            if len(cd):
                #min count filtering
                # q75=np.quantile(cd["count"],0.75)
                # if q75>min_count: cd=cd[cd["count"]>=min_count]
                # else: print("Minimum count too high, skipping filtering!")        
                    
                
                ut=np.unique(cd[col])
                uz=np.zeros(cd[col].max()+1,dtype=np.uint32)
                uz[ut]=np.arange(len(ut))
                
                xys=cd[gcols].values
                split_at=np.argwhere(np.any(np.diff(xys,axis=0)!=0,axis=1))[:,0]
                sxys=np.vstack([xys[split_at],xys[-1]])
                
                if len(sxys)==1: 
                    print("Skipping ROI: too little pixels/scans for MVA")
                    continue
                
                #coo based sparse construction (try to fix this later!)
                # zix=np.zeros((xys.max()+1),dtype=np.int32)
                # zix[sxys]=np.arange(len(sxys)).reshape(-1,1)
                # cd["x"]=zix[cd[gcols].values]
                # cd["y"]=uz[cd[col]]
                # szm=coo_matrix((cd["count"], (cd["x"], cd["y"])), shape=(int(cd.x.max()+1), int(cd.y.max()+1))).tocsr()
            

                #indexing based sparse construction
                sx=np.array_split(cd[[col,"count"]].values.astype(np.uint32),split_at+1)
                szm=np.zeros([len(sxys),len(ut)],dtype=bits(cd["count"].max()))  
                for ix,i in enumerate(sx): szm[ix,uz[i[:,0]]]=i[:,1]
                szm=csr_matrix(szm)
                
                
                # if  Top_quantile: #only keep top abundant  
                #     szm[:,np.argwhere(szm.mean(axis=0)<np.quantile(szm.mean(axis=0),Top_quantile))]=0 
                
                if Remove_zeros: 
                    q=np.asarray(szm.sum(axis=0)>0).flatten()
                    ut=ut[q]
                    szm=szm[:,q]
            
                szmsum=np.array(szm.sum(axis=0)).T
   
                
                ### MVA Scaling 
                
                #not spare first
                if MVA_scaling=="Standard":  szm=StandardScaler(with_mean=False).fit_transform(szm)
                if MVA_scaling=="Robust":    szm=robust_scale(szm,axis=0,with_centering=False)
                if MVA_scaling=="Poisson":   szm=szm/np.sqrt(szm.mean(axis=0))  #2D poisson scaling X/ sqrt mean  /np.sqrt(szm.mean(axis=1))
                if MVA_scaling=="Jaccard":   szm=szm.astype(bool).astype(int)
                szm=szm.T
                if MVA_scaling=="MinMax":   scale_sparse_matrix_rows(szm, lowval=0, highval=1)  #szm=maxabs_scale(szm,axis=0)  #szm=MaxAbsScaler().fit_transform(szm) #doesnt work#szm=MaxAbsScaler().fit_transform(szm)
            
            
                if szm.shape[1]<=MVA_components: n_components=szm.shape[1]-1 
                else:                            n_components=MVA_components
            
                
                for MVA_method in MVA_methods:
                    print(MVA_method)
                     
                    if MVA_method=="PCA":
                        clf = TruncatedSVD(n_components=n_components,algorithm=PCA_algorithm) #random is fast, arpack is more accurate
                        MVA = clf.fit_transform(szm.astype(float))
                        if Varimax: MVA =varimax(MVA) #update !!!
                        loadings=clf.components_#* np.sqrt(clf.explained_variance_).reshape(-1,1)
                
                    if MVA_method=="NMF":
                        model = NMF(n_components=n_components, init=NMF_algorithm, random_state=0, verbose=True,max_iter=20000)
                        MVA = model.fit_transform(szm)
                        loadings=model.components_
                        
                    #reconstruct intensity
                    if reconstruct_intensity:

                        if MVA_method=="PCA":                        
                            n, p = szm.shape
                            p_sums = loadings.sum(axis=1)   # shape (k,)
                            Cik = np.zeros((n, n_components))
                            for k in range(n_components):   Cik[:, k] = MVA[:, k] * p_sums[k]
                        
                        if MVA_method=="NMF":
                            X_hat = MVA @ loadings   
                            Cik = np.zeros((szm.shape[0], n_components))
                            for k in range(n_components): Cik[:, k] = MVA[:, k] * loadings[k, :].sum()
                        
                        global_contribution = Cik.sum(axis=0) 
                        
                        #rescale to total intensity
                        MVA*=global_contribution
                        loadings/=global_contribution.reshape(-1,1)

                
                    ldf=pd.DataFrame(np.hstack([sxys,loadings.T]),columns=gcols+np.arange(n_components).tolist())
#%%
                    mva_mass=c2m(ut,sf,k0,bin_tof=MVA_bin_tof).reshape(-1,1)
                    mva_mass=mva_mass*(1-np.interp(mva_mass,xcal,ycal)/1e6)
                    mdf=pd.DataFrame(np.hstack([mva_mass,MVA]),columns=["mass"]+np.arange(n_components).tolist()) 
                    
                    if data_reduction=="binning": #group by peak
                        
                        
              
                            
                            
                            #%%
                            ls=loadings.sum(axis=1)
                            for nc in range(n_components):
                                print("component: "+str(nc))
                                
                                                                
                                spec=np.zeros(ut.max()+1)
                                spec[ut]=mdf[nc]
                                specs,specpols=[],[]
     
                                sSpectrum,ps,pns=pick_peaks(abs(spec)) #make this abs?                                                                     #this only picks "single peaks"
                                centroids=np.array([centroid(sSpectrum[i[1]:i[2]].index,sSpectrum[i[1]:i[2]].values) for i in ps])   #calculate peak centroids
                                qfin=np.isfinite(centroids)
                                centroids=centroids[qfin]
                                
                                lsf,lk0,lxcal,lycal=CalibrateGlobal(centroids,calibrants,
                                                                    sf,k0,bin_tof=MVA_bin_tof,
                                                                    weights=sSpectrum[ps[qfin,0]].values,
                                                                    filename_add="_"+MVA_method+"_"+str(nc)+"_")
                                #add bin tof argument
                      
                
                
                                
                                #Output per component
                                if Peak_deconvolution:
                                        
    
                                    
                                    #separate in positive and negative (for PCA)
                                    if np.any(spec<0): 
                                        s=spec.copy()
                                        s[s>0]=0
                                        specs.append(s)
                                        specpols.append("neg")
                                    
                                    if np.any(spec>0):
                                        s=spec.copy()
                                        s[s<0]=0
                                        specs.append(s)
                                        specpols.append("pos")
                                    
                                    comb=[]
                                    for sx,spectrum in enumerate(specs):
                                        if specpols[sx]=="neg": spectrum*=-1
    
                                        if reconstruct_intensity: p,pi=find_peaks(spectrum,distance=10,prominence=5)
                                        else:                     p,pi=find_peaks(spectrum,distance=10,prominence=0)
                                            
                                            
                                        pborders=np.vstack([pi["left_bases"],pi["right_bases"]]).T
                                        
                
                                        deconv_peaks=deconvolve(pborders,spectrum,bin_tof=MVA_bin_tof,ext=10,gau_filt=5)
                                        deconv_peaks=np.vstack(deconv_peaks) #peak, #height
                                    
                                    
                                    
                                    
                 
                                        MVApeaks=pd.DataFrame(deconv_peaks,columns=["TOF","Apex"]) 
                                        
                                    
                                    if specpols[sx]=="neg": MVApeaks["Apex"]*=-1
                                    MVApeaks["FWHM"]=np.interp(MVApeaks["TOF"],xres,pres)
                                    MVApeaks["mass"]=c2m((MVApeaks["TOF"]+int(MVA_bin_tof/2))*MVA_bin_tof,sf,k0)
                                    MVApeaks["mass"]*=(1-np.interp(MVApeaks["mass"],xcal,ycal)/1e6)
                                    comb.append(MVApeaks)
                                    
                                MVApeaks=pd.concat(comb).sort_values(by="mass")
        
                                if ROI_c>1:  MVApeaks.to_csv(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_"+str(nc)+"_peaks.csv")
                                else:        MVApeaks.to_csv(fs+str(MVA_dimension)+"D_"+MVA_method+"_"+str(nc)+"_peaks.csv")
                            
        
                                

                        
   
 
                     
                    #recalibrate peak_mass outputs

                    
                    
                    if ROI_c>1: #save with ROI
                        ldf.to_csv(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_loadings.csv")
                        mdf.to_csv(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.csv") 
                    else:
                        ldf.to_csv(fs+"_"+str(MVA_dimension)+"D_"+MVA_method+"_loadings.csv")
                        mdf.to_csv(fs+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.csv")
                        
             
                    ### plot mass loadings
                    rows,cols=n_components//5,5
                    if n_components%5: rows+=1
                    if cols>n_components: cols=n_components
                    
                    fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                    if (rows*cols)==1: axes=np.array([axes])
                    axes = axes.flatten()
                    for i in range(n_components):
                        axes[i].plot(c2m(ut*MVA_bin_tof,sf,k0),MVA[:,i])
                    
                    fig.supylabel('Loading')
                    fig.supxlabel('mz')
                    
                    
                    
                    for ix,i in enumerate(axes):
                        i.title.set_text(ix+1)
                    
                    if ROI_c>1:        fig.savefig(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.png",dpi=300,bbox_inches="tight") #save summed intensity png
                    else:              fig.savefig(fs+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.png",dpi=300,bbox_inches="tight") #save summed intensity png     
                    plt.close()     
                    ###### Spatial plots
                    
                
              
                    #% Plot 1D
                    
                    if MVA_dimension==1:
                        
                        rows,cols=n_components//5,5
                        if n_components%5: rows+=1
                        if cols>n_components: cols=n_components
                        
                        fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                        if (rows*cols)==1: axes=np.array([axes])
                        axes = axes.flatten()
                        for i in range(n_components):
                            
                            axes[i].plot(sxys*sputtertime/scans,loadings[i,:])
                            # axes[i].set_xlabel('mz')
                            # if not i%5: axes[i].set_ylabel('loading') 
                            
                        for ix,i in enumerate(axes):
                            i.title.set_text(ix)
                            
                        fig.supylabel('Loading')
                        fig.supxlabel('Sputter time [s]')
                        
                        if ROI_clusters>1: fig.savefig(fs+"_ROI"+str(roi)+"_"+MVA_method+"_1D_loadings.png",dpi=300,bbox_inches="tight") #save summed intensity png
                        else:              fig.savefig(fs+"_"+MVA_method+"_1D_loadings.png",dpi=300,bbox_inches="tight") #save summed intensity png
                        plt.close()
  
                    
                    #% Plot 2D
                    
                    if MVA_dimension==2:
                    
                        rows,cols=n_components//5,5
                        if n_components%5: rows+=1
                        if cols>n_components: cols=n_components
                        
                        fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                        if (rows*cols)==1: axes=np.array([axes])
                        axes = axes.flatten()
                        for i in range(n_components):
                            
                            hm=pd.DataFrame(np.hstack([sxys,loadings[i,:].reshape(-1,1)]),columns=["x","y","c"]).pivot(columns="x",index="y").fillna(0)
                            hm.columns = hm.columns.droplevel().astype(int)
                            hm.index=hm.index.astype(int)
                            
                            g=sns.heatmap(hm,ax=axes[i],cbar=False,center=0,robust=True,cmap='RdBu_r')
                            
                            #update to pixels to micrometers
                            axes[i].set_yticklabels((np.linspace(0,y_um,len(g.get_yticks()))).astype(int),rotation=0)
                            if not i%5: axes[i].set_ylabel(r'y $\mu$m') 
                            else:  axes[i].set_ylabel('') #prevent bleed
                            
                            axes[i].set_xticklabels((np.linspace(0,x_um,len(g.get_xticks()))).astype(int),rotation=90)
                            axes[i].set_xlabel(r'x $\mu$m')
                            
                        for ix,i in enumerate(axes):
                            i.title.set_text(ix)
                        
                        fig.savefig(fs+"_"+MVA_method+"_2D_loadings.png",dpi=300,bbox_inches="tight") #save summed intensity png
                        plt.close()
                        

                    
                    #% Plot 3D
                    
                    if MVA_dimension==3:
                    
         
                        cmax=max(loadings.max(),abs(loadings.min()))
                        #### Surface plot
                        
                        for i in range(n_components):
                       
                            vdf=pd.DataFrame(np.vstack([sxys.T,loadings[i,:].reshape(1,-1)]).T,columns=["x","y","scan","counts"])
                        
                            #X,Y,Z=np.meshgrid(np.arange(math.ceil(xpix)), np.arange(math.ceil(ypix)),np.arange(math.ceil(scans)))
                            xpx,ypx,scx= xpix/MVA_bin_pixels,ypix/MVA_bin_pixels,scans/MVA_bin_scans
                            
                            Y,X,Z=np.meshgrid(*[np.arange(i) for i in [int(vdf.y.max()+1),
                                                                       int(vdf.x.max()+1),
                                                                       int(vdf.scan.max()+1)]])
                            # X,Y,Z=np.meshgrid(np.arange(int(vdf.x.max()+1)), 
                            #                   np.arange(int(vdf.y.max()+1)),
                            #                   np.arange(int(vdf.scan.max()+1)))
                            
                            
                            vm=np.zeros([math.ceil(xpx),math.ceil(ypx),math.ceil(scx)])
                            vm[vdf["x"].astype(int).tolist(),vdf["y"].astype(int).tolist(),vdf["scan"].astype(int).tolist()]=vdf["counts"].values
                    
                            
                            x,y,z,v=X.flatten(),Y.flatten(),Z.flatten(),vm.flatten()
                            
                            #remove background points
                            q=np.argwhere(abs(v)>=(np.quantile(abs(v),0.90)/10))
              
                            
                            
                            
                            x,y=x*x_um/xpx,y*y_um/ypx #convert to micrometer
                            z=z*sputtertime/scx       #convert to sputter time in seconds
                           
                            #here you can interpolate more points for the volume plot if needed
                            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html            
                            
                            fig = go.Figure(data=go.Volume(
                                x=x, y=y, z=z, value=v, 
                               
                                cmax=cmax*0.75,
                                cmin=-cmax*0.75,
                                cmid=0,
                                opacity=0.2, # needs to be small to see through all surfaces
                                surface_count=5, # needs to be a large number for good volume rendering
                                
                                
                                
                                ))
                            fig.update_scenes(zaxis_autorange="reversed")
                            
                            
                            ax_style = dict(showbackground =True,
                                        backgroundcolor="white",
                                        showgrid=False,
                                        zeroline=False)
                        
                        
                        
                            fig.update_layout(template="none", width=600, height=600, font_size=11,
                                              scene=dict(xaxis=ax_style, 
                                                         yaxis=ax_style, 
                                                         zaxis=ax_style,
                                                         camera_eye=dict(x=1.85, y=1.85, z=1)))
                            
                            
                            fig.update_layout(scene = dict(
                                      xaxis=dict(title=dict(text='y [micrometer]')),
                                      yaxis=dict(title=dict(text='x [micrometer]')),
                                      zaxis=dict(title=dict(text='Sputter time [s]')),),
                                    width=700,margin=dict(r=20, b=10, l=10, t=10))
                                
                            camera = dict(
                                up=dict(x=0, y=0, z=1),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=1.65, y=1.25, z=1.55)
                            )
                    
                    
                            fig.update_layout(scene_camera=camera, title="Component: "+str(i))
                            fig.update_scenes(aspectmode="cube")
                            
                            
                            #fig.show() #no fig show but write directly
                            fig.write_html(fs+"_"+MVA_method+"_3D_comp"+str(i)+"loading.html") #save summed intensity 
                    
                    
      
                    
                 


