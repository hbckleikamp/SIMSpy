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

import itertools


from skmisc import loess #specific module for weighted Loess

from scipy.optimize import curve_fit, nnls
from scipy.signal import find_peaks,find_peaks_cwt, savgol_filter
from scipy.sparse import csr_matrix, coo_matrix
from scipy.ndimage import gaussian_filter

from sklearn.preprocessing import robust_scale, maxabs_scale, StandardScaler #,MaxAbsScaler
from sklearn.decomposition import PCA,TruncatedSVD,NMF
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn.metrics import r2_score
from sklearn.neighbors import KDTree

import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize

    

    

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



itmfiles=[
    
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot I_A2.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_A1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot II_B1.itm", #negative (misnamed)

"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot I_B1.itm", 
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot II_A1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_B1.itm",
"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_C1(sputter).itm"

# Fast-imaging didnt work propely
# "E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_A2(images).itm",
#"E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/P_Spot VI_C1(images).itm",
# "E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_A2(images).itm",
# "E:/Data/TOF_SIMS/UCL/110825_Sheaths/20250812_Cable bacteria/N_Spot VI_C1(images).itm",

]




grd_exe="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe" #path to Grd executable 
Output_folder=""
write_params=True
load_params=""

#initial dimension redution
max_mass=200         #truncate mass
max_scans=False      #truncate scans
min_scans=20 #050
min_width_ratio=0 #0.1  #remove based on ratio to expected peakwidth
max_width_ratio=0 #5    #remove based on ratio to expected peakwidth
Remove_bin_edges=True   #when corners do not satifsy complete bins, trim them

#peak picking
prominence=10
distance=20       #minimum distance between peaks
extend_window=20  #extend around peak bases
cwt_w=10          #wavelet transform window

#Calibration
ppm_cal=200          #maximum deviation for finding internal calibrants
Substrate="InOSiSn"         #list of elements that are present on substrate
Substrate_Calibrants=str(Path(basedir,"Substrate_Calibrants.csv"))  #list of ions coming from substrate
Calibrants=str(Path(basedir,"Calibrants.csv"))                      #list of typical background ions

#Target parameters
Targets=str(Path(basedir,"Targets.csv"))
ppm=100
Target_bin_pixels=2               
Target_bin_scans=2
Target_dimension=2

#plotting
Sum_by_Groups=True #plot 1/2/3D map for each combined fragment summed over groups
Plot_combined=True #plot combined depth profiles (it Target_dimension =1)
tile_plots=True    #combine plots in tiled output
row_entries=4      #number of rows in tile plot

#untargeted group detection
Determine_groups=True #Cluster group components
Scaling="MinMax" #Scaling for PCA Standard Robust Poisson MinMax Jaccard
n_components=3   #number of components in PCA
cluster_distance="auto" #0.6 #Cosine distance
Expand_groups=0  #add non-target peaks to groups (only if Determine_groups ==True)

#Pairwise comparison
pairwise_diff=True #compute distance between groups
maximize_difference=True
correlate=True




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

    #Target parameters
    parser.add_argument("--Targets", required = False, default=str(Path(basedir,"Targets.csv")), help="Target fragments to exctract")
    parser.add_argument("--ppm", required = False, default=100, help="ppm mass error tolerance for finding target fragments")
    parser.add_argument("--Target_bin_pixels", required = False, default=2, help="Data reduction by binning pixels")
    parser.add_argument("--Target_bin_scans", required = False, default=2, help="Data reduction by binning scans")
    parser.add_argument("--Target_dimension", required = False, default=2, help="How many dimensions to analyze (1,2,3)")

    #plotting
    parser.add_argument("--Sum_by_Groups", required = False, default=True, help="plot 1/2/3D map for each combined fragment summed over groups")
    parser.add_argument("--Plot_combined", required = False, default=True, help="plot combined depth profiles (if Target_dimension =1)")
    parser.add_argument("--tile_plots", required = False, default=True, help="combine plots in tiled output")
    parser.add_argument("--row_entriesn", required = False, default=4, help="number of rows in tile plot")
    
    #untargeted group detection
    parser.add_argument("--Determine_groups", required = False, default=str(Path(basedir,"Targets.csv")), help="Toggle untargeted group detection")
    parser.add_argument("--Scaling", required = False, default="MinMax", help="Scaling for PCA Standard Robust Poisson MinMax Jaccard")
    parser.add_argument("--n_components", required = False, default=3, help="number of components in PCA")
    parser.add_argument("--cluster_distance", required = False, default="auto", help="Cosine distance")
    parser.add_argument("--Expand_groups", required = False, default=50, help="add non-target peaks to groups (only if Determine_groups ==True)")
    
    #Pairwise comparison
    parser.add_argument("--pairwise_diff", required = False, default=True, help="compute abundance difference between groups")
    parser.add_argument("--maximize_difference", required = False, default=True, help="scale groups for maximum difference")
    parser.add_argument("--correlate", required = False, default=True, help="plot cosine correlation between groups")
    
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
def c2m(c,sf,k0):    return ((c - k0) / (sf)) ** 2   
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
 



def Scaler(x,Scaling="Standard"): #Standard Robust Poisson MInMax Jaccard

    if Scaling=="Standard": x=(x-np.mean(x))/x.std()
    if Scaling=="Robust":   x=robust_scale(x)
    if Scaling=="Poisson":  x=x/np.sqrt(x.mean())  #2D poisson scaling X/ sqrt mean  /np.sqrt(szm.mean(axis=1))
    if Scaling=="Jaccard":  x=x.astype(bool).astype(int)
    if Scaling=="MinMax":   x=(x - x.min()) / (x.max() - x.min()) #szm=maxabs_scale(szm,axis=0)  #szm=MaxAbsScaler().fit_transform(szm) #doesnt work#szm=MaxAbsScaler().fit_transform(szm)
    return x


def CalibrateGlobal(channels,calibrants,sf,k0,
                    ppm_cal=ppm_cal,
                    min_mass=10,
                    plot=True,
                    weights=""):
        

    try: 
            
        #%%
   
        # # #test
        # sf,k0=I.sf,I.k0
        # channels=centroids
        # ppm_cal=200
        # min_mass=10
        # plot=True
        # weights=sSpectrum[ps[:,0]].values

        

        fcalibrants=calibrants[calibrants>min_mass]
        q=np.argsort(channels)
        channels=channels[q]
        if len(weights): weights=weights[q]
        mp=c2m(channels,sf,k0)
        
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
        x,y=caldf["channels"],caldf["c"]
        y_pred=c2m(x,sf,k0)

        
        
        ### base recalibration if fit is bad

        if len(weights): r2 = r2_score(y, y_pred, sample_weight=caldf["intensity"].values)
        else:            r2 = r2_score(y, y_pred)
        if r2<0.95: 
            if len(weights): [sf,k0], _ = curve_fit(c2m,x,y,p0=[sf,k0],sigma=1/caldf["intensity"].values, absolute_sigma=True)
            else:            [sf,k0], _ = curve_fit(c2m,x,y,p0=[sf,k0])
            
            caldf["mass"]=c2m(caldf["channels"],sf,k0)
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
            fig.savefig(fs+"_glob_cal_scat.png",bbox_inches="tight",dpi=300)
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
            fig.savefig(fs+"_glob_cal_hist.png",bbox_inches="tight",dpi=300)
            plt.close()
            
#%%
    except:
       
        print("calibration failed!, increase calibration ppm?")
        return I.sf,I.k0,c2m(centroids,sf,k0),np.array([0]*len(centroids))
        
    pd.DataFrame([[sf,k0]],columns=["sf","k0"]).to_csv(fs+"calib.csv")

    return sf,k0,xq,pq


def smooth_spectrum(ds):
    
    Spectrum=np.bincount(ds["tof"].values)
       
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


def Plot3D(cmat,name,denoise=False,surface_count=5):
    
    if denoise>0:
        ab=abs(cmat)
        cmat[ab<np.quantile(cmat,denoise)]=0
    cmax=cmat.max()
    X,Y,Z=np.meshgrid(*[np.arange(i) for i in cmat.shape])
    x,y,z,v=X.flatten(),Y.flatten(),Z.flatten(),cmat.flatten()
    x,y=x*x_um/xpix,y*y_um/ypix #convert to micrometer
    z=z*sputtertime/scans       #convert to sputter time in seconds
   
    #here you can interpolate more points for the volume plot if needed
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html            
    
    fig = go.Figure(data=go.Volume(
        x=x, y=y, z=z, value=v, 
       
        cmax=cmax*0.75,
        cmin=cmat.min(),
        cmid=cmat.mean(),
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=surface_count #10 #20, #5 needs to be a large number for good volume rendering
        
        
        
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

    fig.update_layout(scene_camera=camera, title=name)
    fig.update_scenes(aspectmode="cube")
    
    
    fig.show() #no fig show but write directly
    fig.write_html(fs+"_3D_"+str(name)+".html")


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
    
    print(itmfile)
    I = pySPM.ITM(itmfile)
    
    ### Get ITM metadata ###
    xpix,ypix= math.ceil(I.size["pixels"]["x"]), math.ceil(I.size["pixels"]["y"])
    scans=math.ceil(I.Nscan)
    k0,sf=I.k0,I.sf

    mode_sign={"positive":"+","negative":"-"}.get((I.polarity).lower()) 


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
        
    
    #truncate scans
    if max_scans>0: ds=ds[ds["scan"]<=max_scans] 
    if min_scans>0: ds=ds[ds["scan"]>=min_scans]
    
    #trim bin edges
    if Remove_bin_edges:
        if Target_bin_pixels: xpix-=xpix%Target_bin_pixels
        if Target_bin_pixels: ypix-=ypix%Target_bin_pixels
        if Target_bin_scans: scans-=scans%Target_bin_scans
        ds= ds[ (ds.x<xpix)  & (ds.y<ypix)  & (ds.scan<scans)   ]
    
    #binning
    ds.scan=(ds.scan/Target_bin_scans).astype(int)
    ds[["x","y"]]=( ds[["x","y"]]/Target_bin_pixels).astype(int)
    scans,xpix,ypix=int(round(scans/Target_bin_scans,0)),int(round(xpix/Target_bin_pixels,0)),int(round(ypix/Target_bin_pixels,0))


    calibrants=np.sort(np.array([pySPM.utils.get_mass(i) for i in Calibrants if i.endswith(mode_sign)])) #shortended calibrants list
    x_um,y_um=I.size["real"]["x"]*1e6,I.size["real"]["y"]*1e6
    sputtertime=I.get_value("Measurement.SputterTime")["float"] #not sure if correct
        
    ###### Global calibration ######
    
    
    
    sSpectrum,ps,pns=pick_peaks(ds) #this only picks "single peaks" 
    np.save(fs+"_summed_spectrum",sSpectrum) #save summed spectrum
                                                                    
    peaks=np.vstack([ps,pns])
    lb,rb=peaks[:,1],peaks[:,2]
    ds=ds[np.in1d(ds.tof,create_ranges(np.vstack([lb,rb]).T))]
    
    centroids=np.array([centroid(sSpectrum[i[1]:i[2]].index,sSpectrum[i[1]:i[2]].values) for i in ps])   #calculate peak centroids
    print("Calibrating")
    sf,k0,xcal,ycal=CalibrateGlobal(centroids,calibrants,sf,k0,ppm_cal=ppm_cal,weights=sSpectrum[ps[:,0]].values) #calibrate

    #truncate mass
    if max_mass: ds=ds[c2m(ds["tof"],sf,k0)<max_mass] 
    #%% gussian filter of good hits here
    
    if min_width_ratio or max_width_ratio:
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

    
        # ###### Filter channels on peakwidth ###### 
          
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
    

    ###### Assign peaks ######
    
    pmat=np.zeros(peaks.max()+1,dtype=np.int64)
    pmat[create_ranges(peaks[:,1:])]=np.repeat(np.arange(len(peaks)),peaks[:,2]-peaks[:,1])
    ds["peak_bin"]=peaks[pmat[ds["tof"]],0]

    peak_masses=c2m(peaks[:,0],sf,k0)
    peak_masses*=(1-np.interp(peak_masses,xcal,ycal)/1e6) #correct ppm
        
    #%%    Extract targets
    

    
    tar=pd.read_csv(Targets)
    tar=tar[tar.Formula.str.endswith(mode_sign)]
    tar=tar[tar.Group.isin(["Protein","Glycan"])].reset_index(drop=True) #testing
    tarmass=np.hstack([getMz(i) for i in tar.Formula])
    
    tree = KDTree(peak_masses.reshape(-1,1), leaf_size=200) 
    l=tree.query_radius(tarmass.reshape(-1,1), r=tarmass*ppm/1e6)
    links=np.vstack([np.vstack([np.repeat(ix,len(i)),i]).T for ix,i in enumerate(l) if len(i)])
    
    ldf=pd.DataFrame(links,columns=["tx","px"])
    ldf["peak_bin"]=peaks[ldf.px,0]
    ldf[["Formula","Group"]]=tar.iloc[ldf.tx].values
    ldf=ldf[["peak_bin", "Formula", "Group"]].set_index("peak_bin")
    ldf["Mass"]=pd.concat([getMz(i) for i in ldf.Formula]).values

    
    tards=ds.loc[ds.peak_bin.isin(ldf.index),["scan","x","y","peak_bin"]]

    if Determine_groups:
        
        ldf=ldf[["Formula","Mass"]].drop_duplicates()

        gcols=["x","y"] #["scan","x","y"]
        cd=tards.groupby(gcols+["peak_bin"]).size().to_frame('count').reset_index()
        ut=np.unique(cd["peak_bin"])
        uz=np.zeros(cd["peak_bin"].max()+1,dtype=np.uint32)
        uz[ut]=np.arange(len(ut))
        
        szm=coo_matrix((cd["count"], (cd["x"]*xpix+cd["y"], uz[cd["peak_bin"]])), shape=(xpix*ypix, len(ut))).tocsr().T
        #scale_sparse_matrix_rows(szm) #this guy is not working
        #pca = TruncatedSVD(n_components=n_components,algorithm="arpack" ).fit(szm) 
        
        #non-sparse
        szm=np.array(szm.todense())
        szm=( szm-szm.min(axis=1).reshape(-1,1)) / (szm.min(axis=1)-szm.max(axis=1)).reshape(-1,1) #minmax scale
        pca = PCA(n_components=n_components).fit(szm)
        
        
        


        
        X_reduced = pca.transform(szm)
        loadings=pca.components_.T*np.sqrt(pca.explained_variance)
        #loadings=pca.components_.T
        
        sxys=np.vstack([i for i in itertools.product(np.arange(xpix),np.arange(ypix))])
        ### % plot loadings
        rows,cols=n_components//5,5
        if n_components%5: rows+=1
        if cols>n_components: cols=n_components
        
        fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
        if (rows*cols)==1: axes=np.array([axes])
        axes = axes.flatten()
        
        for i in range(n_components):
    
            hm=pd.DataFrame(np.hstack([sxys,loadings[:,i].reshape(-1,1)]),columns=["x","y","c"]).pivot(columns="x",index="y").fillna(0)
            hm.columns = hm.columns.droplevel().astype(int)
            hm.index=hm.index.astype(int)
            
            g=sns.heatmap(hm,ax=axes[i],cbar=False,center=0,robust=True)
            
            #update to pixels to micrometers
            axes[i].set_yticklabels((np.linspace(0,y_um,len(g.get_yticks()))).astype(int),rotation=0)
            if not i%5: axes[i].set_ylabel(r'y $\mu$m') 
            else:  axes[i].set_ylabel('') #prevent bleed
            
            axes[i].set_xticklabels((np.linspace(0,x_um,len(g.get_xticks()))).astype(int),rotation=90)
            axes[i].set_xlabel(r'x $\mu$m')
            
        for ix,i in enumerate(axes):
            i.title.set_text(ix+1)
            
        fig.savefig(fs+"_groups_PCA.png",dpi=300)
            
        
        #%% add new
        if Expand_groups>0:
            
            speaks=ds.groupby("peak_bin").size().sort_values(ascending=False)
            top_peaks=speaks[~speaks.index.isin(ldf.index)].index[:Expand_groups]
  
            cd=ds[ds["peak_bin"].isin(top_peaks)].groupby(gcols+["peak_bin"]).size().to_frame('count').reset_index()
            ut_new=np.unique(cd["peak_bin"])
            uz=np.zeros(cd["peak_bin"].max()+1,dtype=np.uint32)
            uz[ut_new]=np.arange(len(ut_new))
            
            szm=coo_matrix((cd["count"], (cd["x"]*xpix+cd["y"], uz[cd["peak_bin"]])), shape=(xpix*ypix, len(ut_new))).tocsr().T
            
            # #sparse
            # scale_sparse_matrix_rows(szm)
            
            #non-sparse
            szm=np.array(szm.todense())
            szm=( szm-szm.min(axis=1).reshape(-1,1)) / (szm.min(axis=1)-szm.max(axis=1)).reshape(-1,1) #minmax scale

            
            X_new_pca = pca.transform(szm)      # Project onto the same PCs
            
            #update old values
            X_reduced=np.vstack([X_reduced,X_new_pca])
            ut=np.hstack([ut,ut_new])
            ldf=pd.concat([ldf,pd.DataFrame(np.vstack([np.round(c2m(ut_new,sf,k0),2),
                                                       c2m(ut_new,sf,k0)]).T,
                                            index=ut_new,columns=["Formula","Mass"])])

        #%% cluster groups

        # try:
        d = sch.distance.pdist(X_reduced,metric="cosine")
        L = sch.linkage(d, method='average')
 
        if cluster_distance=="auto": clustd=0.3*d.max()
        else:                        clustd=cluster_distance
 
        clusters = sch.fcluster(L, clustd, 'distance')
        
        #deal with non-unique peak bins
        ldf=ldf.loc[ut]
        ldf=ldf.groupby(ldf.index)["Formula"].apply(lambda x: ", ".join(list(set([str(i) for i in x])))).to_frame("Formula")

        cdf=pd.DataFrame(np.vstack([np.unique(ut),clusters-1]).T,columns=["peak_bin","Group"]).set_index("peak_bin")
        ldf=ldf.merge(cdf,left_index=True,right_index=True)



        
        fig,ax=plt.subplots()
        dendrogram = hierarchy.dendrogram(L)
        plt.axhline(y=clustd, c='k',linestyle="--")
        plt.legend(["Clusters","Cutoff"])
        fig.savefig(fs+"_groups_dendrogram.png",dpi=300)
     
        # except:
        #     print("Warning! couldn't cluster")

    
    
        ccol,groups="peak_bin",ldf.Formula.values
        tards=ds.loc[ds.peak_bin.isin(ldf.index),["scan","x","y","peak_bin"]]
    
    #save updated ldf
    ldf["mMass"]=c2m(ldf.index,sf,k0)
    ldf["mMass"]*=(1-np.interp(ldf["mMass"],xcal,ycal)/1e6) #correct ppm
    ldf["Count"]=tards.groupby("peak_bin").size()   
    
    q=ldf["Formula"].str.contains(mode_sign,regex=False)
    ldf.loc[q,"pMass"]=[getMz(i).values[0] for i in ldf.loc[q,"Formula"].values]
    ldf.loc[q,"ppm"]=(ldf.loc[q,"pMass"]-ldf.loc[q,"mMass"])/ldf.loc[q,"pMass"]*1e6
    
    
    ldf=ldf.sort_values(by=["Group","mMass"])
    ldf.to_csv(fs+"groups.csv")
    
    
    
    
        
    if Sum_by_Groups:   
        tards["Group"]=ldf.loc[tards.peak_bin,"Group"].values
        ccol="Group"
        groups=np.unique(ldf["Group"])  
    
    
    pairs=[]
    
    #bin pixels, bin scans?
    if Target_dimension==1: gcols,zmat=["scan"],         np.zeros(scans,dtype=np.uint32) 
    if Target_dimension==2: gcols,zmat=["x","y"],        np.zeros((xpix,ypix),dtype=np.uint32) 
    if Target_dimension==3: gcols,zmat=["x","y","scan"], np.zeros((xpix,ypix,scans),dtype=np.uint32) 
    
    
    #loop over dimensions
    utards=tards.groupby(gcols+[ccol]).size().to_frame("c").reset_index()
    

    

    #%% 1D
    
    
    if Target_dimension==1:
    

        if Plot_combined: 
            
            if tile_plots:
                ng=len(groups)
                rows,cols=ng//row_entries,row_entries
                if ng%row_entries: rows+=1
                if cols>ng: cols=ng
                fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                if (rows*cols)==1: axes=np.array([axes])
                axes = axes.flatten()
            
            else:
                fig,ax=plt.subplots()
        
        
        for i,[n,g] in enumerate(utards.groupby(ccol)):
            
            if not Sum_by_Groups: n=ldf.loc[n,"Formula"]
            
            cmat=zmat.copy()
            cmat[g.scan]=g.c
            cmat=Scaler(cmat,Scaling)
            if pairwise_diff: pairs.append(cmat)
            if not Plot_combined: fig,ax=plt.subplots()
            
            if tile_plots:
                axes[i].plot(np.arange(len(cmat))*sputtertime/scans,cmat)
                if not i%row_entries:axes[i].set_ylabel("Abundance")
                if i>=(rows-1)*cols: axes[i].set_xlabel("Sputter time [s]")
                axes[i].title.set_text(n)
            else:
                plt.plot(np.arange(len(cmat))*sputtertime/scans,cmat)
                plt.ylabel("Abundance")
                plt.xlabel("Sputter time [s]")
                plt.title(n)
                fig.savefig(fs+"_1D_"+n+".png",dpi=300)
      
    
        if Plot_combined: 
            
            if not tile_plots:
                plt.legend(groups)
    
            fig.savefig(fs+"_1D_Targets.png",dpi=300)
            

        
    #%% 2D

    if Target_dimension==2:
        

        if tile_plots:
            ng=len(groups)
            rows,cols=ng//row_entries,row_entries
            if ng%row_entries: rows+=1
            if cols>ng: cols=ng
            fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
            if (rows*cols)==1: axes=np.array([axes])
            axes = axes.flatten()
        
        else:
            fig,ax=plt.subplots()
    
        for i,[n,g] in enumerate(utards.groupby(ccol)):
            
            if not Sum_by_Groups: n=ldf.loc[n,"Formula"]
            
            cmat=zmat.copy()
            cmat[g.x,g.y]=g.c
            cmat=Scaler(cmat,Scaling)
            if pairwise_diff: pairs.append(cmat)
  
            if tile_plots:
                
                h=sns.heatmap(cmat.T,ax=axes[i])
                axes[i].set_xticklabels((np.linspace(0,x_um,len(h.get_xticks()))).astype(int),rotation=90)
                axes[i].set_yticklabels((np.linspace(0,y_um,len(h.get_yticks()))).astype(int),rotation=0)
                if not i%row_entries:axes[i].set_ylabel(r'y $\mu$m') 
                if i>=(rows-1)*cols: axes[i].set_xlabel(r'x $\mu$m')
                axes[i].title.set_text(n)
            else:
                
                fig,ax=plt.subplots()
                h=sns.heatmap(cmat.T)
                
                ax.set_xticklabels((np.linspace(0,x_um,len(h.get_xticks()))).astype(int),rotation=90)
                ax.set_yticklabels((np.linspace(0,y_um,len(h.get_yticks()))).astype(int),rotation=0)
                plt.ylabel(r'y $\mu$m')
                plt.xlabel(r'x $\mu$m')
                plt.title(n)
                fig.savefig(fs+"_2D_"+str(n)+".png",dpi=300)
      
        if tile_plots: fig.savefig(fs+"_2D_Targets.png",dpi=300) 
      
  

  #%% 3D

    
    if Target_dimension==3:
    

        for n,g in utards.groupby(ccol):
    
            
            
            
            print(n)
            
            cmat=zmat.copy()
            cmat[g.x,g.y,g.scan]=g.c
            cmat=Scaler(cmat,Scaling)   
            if pairwise_diff: pairs.append(cmat)
            
            if Sum_by_Groups: name=n
            else: name="_".join(ldf.loc[ldf.peak_bin==n,"Formula"].values.tolist())
            
            Plot3D(cmat,name)
            
  
            
  
    
    #%%  Pairwise correlation
  


    combs=[i for i in itertools.combinations(np.arange(len(pairs)),2)]

    
  
    

    
    
    if tile_plots:
        rows,cols=len(pairs)-1,len(pairs)-1



        
        #diff plot
        fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
        if (rows*cols)==1: axes=np.array([axes])
        axes = axes.flatten()
        
        #correlation plot
        

    cormat=np.diag(np.ones(len(pairs)))
    for comb in combs:
        i,j=comb
    
        data1,data2=pairs[i],pairs[j]
        n=str(groups[i])+" - "+str(groups[j])
     
   
        if correlate:
            a,b=data1.flatten(),data2.flatten()
            cormat[i,j] = np.dot(a, b)/(norm(a)*norm(b))            

        
 
        def cost(params):
            scale, offset = params
            return np.sum((data1 - (scale * data2 + offset))**2)
        
        if maximize_difference:
            result = minimize(cost, x0=[1, 0])
            scale, offset = result.x
            data2 = scale * data2 + offset
            
        difference=data1-data2
        
        
        
        if Target_dimension==1:
            
            
            if tile_plots:
                x=i*rows+j-1
                axes[x].plot(np.arange(len(difference))*sputtertime/scans,cmat)
                axes[x].set_xticklabels((np.linspace(0,x_um,len(h.get_xticks()))).astype(int),rotation=90)
                axes[x].set_yticklabels((np.linspace(0,y_um,len(h.get_yticks()))).astype(int),rotation=0)
                if i==j-1:
                    axes[x].set_ylabel('Difference') 
                    axes[x].set_xlabel('Sputter time [s]')
                axes[x].title.set_text(n)
 
                
            else:
                
                fig,ax=plt.subplots()
                plt.plot(np.arange(len(difference))*sputtertime/scans,cmat)
                
                ax.set_xticklabels((np.linspace(0,x_um,len(h.get_xticks()))).astype(int),rotation=90)
                ax.set_yticklabels((np.linspace(0,y_um,len(h.get_yticks()))).astype(int),rotation=0)
                plt.ylabel("Abundance")
                plt.xlabel("Sputter time [s]")
                plt.title(n)
                fig.savefig(fs+"_pairdiff_1D_"+str(n)+".png",dpi=300)
                

          
       
        
        if Target_dimension==2:
    

            if tile_plots:
                x=i*rows+j-1
                h=sns.heatmap(difference.T,ax=axes[x],robust=True)
                axes[x].set_xticklabels((np.linspace(0,x_um,len(h.get_xticks()))).astype(int),rotation=90)
                axes[x].set_yticklabels((np.linspace(0,y_um,len(h.get_yticks()))).astype(int),rotation=0)
                if i==j-1:
                    axes[x].set_ylabel(r'y $\mu$m') 
                    axes[x].set_xlabel(r'x $\mu$m')
                axes[x].title.set_text(n)
            else:
                
                fig,ax=plt.subplots()
                h=sns.heatmap(difference.T,robust=True)
                
                ax.set_xticklabels((np.linspace(0,x_um,len(h.get_xticks()))).astype(int),rotation=90)
                ax.set_yticklabels((np.linspace(0,y_um,len(h.get_yticks()))).astype(int),rotation=0)
                plt.ylabel(r'y $\mu$m')
                plt.xlabel(r'x $\mu$m')
                plt.title(n)
                fig.savefig(fs+"_2D_"+str(n)+".png",dpi=300)
          
       
     
            
        if Target_dimension==3:
      
            Plot3D(difference,
                   name=str(groups[i])+" - "+str(groups[j]),
                   surface_count=5)

    if tile_plots:
        if Target_dimension==1: fig.savefig(fs+"pairdiff_1D.png",dpi=300)
        if Target_dimension==2: fig.savefig(fs+"pairdiff_2D.png",dpi=300)
    
    #%%
    
  
    if correlate:
    
        # try: 
        x, y = np.meshgrid(np.arange(len(cormat)), np.arange(len(cormat)))
        x, y = x.flatten(), y.flatten()
        n=len(cormat)
    
        sizes = np.abs(cormat) * 3000  # Scale to control bubble size
    
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x, y,s=sizes.flatten(),c=cormat,cmap='coolwarm',edgecolor='gray')
        
    
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(np.arange(n), rotation=0, ha='right')
        ax.set_yticklabels(np.arange(n))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_aspect('equal')  # Make cells square
        
        # Add grid lines to emphasize the structure
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)
            
        # Optional: Add correlation text
        for i, (xi, yi, val) in enumerate(zip(x, y, cormat.flatten())):
            ax.text(xi, yi, f"{val:.2f}", ha='center', va='center', fontsize=8, color='black')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Correlation')
        
        plt.tight_layout()
        plt.gca().invert_yaxis()
        fig.savefig(fs+"correlation_"+str(Target_dimension)+"D.png",dpi=300)


        # except:
        #     print("Warning! correlation failed" )
