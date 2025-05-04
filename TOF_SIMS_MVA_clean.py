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
itmfile="" 

grd_exe="C:/Program Files (x86)/ION-TOF/SurfaceLab 6/bin/ITRawExport.exe" #path to Grd executable 
Output_folder=str(Path(basedir,Path(itmfile).stem))

#initial dimension redution
max_mass=1000       #truncate mass
min_width_ratio=0.1 #remove based on ratio to expected peakwidth
max_width_ratio=5   #remove based on ratio to expected peakwidth

#peak picking
prominence=10
distance=20       #minimum distance between peaks
extend_window=20  #extend around peak bases
cwt_w=10          #wavelet transform window

#ROI
ROI_peaks=1000
ROI_clusters=4
ROI_dimensions=3
ROI_bin_pixels=6
ROI_bin_scans=1
ROI_scaling="Jaccard"           # Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"

#depth profile
normalize=False
smoothing=False

#MVA dimension reduction
data_reduction="binning" #"binning" #or peak_picking

#if binning
MVA_bin_tof=5           # bins tof (mass)
MVA_bin_pixels=6       # bins in 2 directions: 2->, 3->9 data reduction
MVA_bin_scans=1         # bin scans
min_count=2         # remove bins with les than x counts

#MVA
MVA_dimensions=[1,2,3]
n_components=5          # number of components
MVA_methods=["NMF","PCA"]
MVA_scaling="MinMax"    # Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"

Top_quantile=0.25 #False      # [0-1, float] only retain mass channels that have top X abundance quantile 
Remove_zeros=True       #
PCA_algorithm="arpack"  # 'arpack' or 'random'
Varimax=False           # perform Varimax rotation in PCA, improves scale of mass components
NMF_algorithm='nndsvd'

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
    parser.add_argument("-o", "--Output_folder", required = False, default=str(Path(basedir,Path(itmfile).stem)), help="Output folder")
    
    #initial dimension redution
    parser.add_argument("--max_mass", required = False, default=1000, help="removes masses above this threshold")
    parser.add_argument("--min_width_ratio", required = False, default=0.1, help="remove masses below the ratio to expected peakwidth")
    parser.add_argument("--max_width_ratio", required = False, default=5, help="remove masses above the ratio to expected peakwidth")
    
    #peak picking
    parser.add_argument("--prominence", required = False, default=10, help="minimum prominence for peak detection")
    parser.add_argument("--distance", required = False, default=20, help="minimum distance for peak detection")
    parser.add_argument("--extend_window", required = False, default=20, help="peak extension")
    parser.add_argument("--cwt_w", required = False, default=10, help="window for continuous wavelet transform peak filtering")

    parser.add_argument("--ROI_clusters", required = False, default=2, help="Regions of interest (ROI) detected")
    parser.add_argument("--ROI_peaks", required = False, default=1000, help="number of peaks considered for ROI detection")
    parser.add_argument("--ROI_dimensions", required = False, default=2, help="ROI dimensions (2 or 3")
    parser.add_argument("--ROI_scaling", required = False, default="Jaccard", help=' Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"')
    parser.add_argument("--ROI_bin_pixels", required = False, default=3, help='if binning: bin pixels in x & y direction')    
    parser.add_argument("--ROI_bin_scans", required = False, default=5, help='if binning: bin frames (scans)')   

    #depth profile
    parser.add_argument("--normalize", required = False, default=False, help=' normalize depth profile to total count')
    parser.add_argument("--smoothing", required = False, default=0, help=' moving average smoothing window')

    #MVA data reduction
    parser.add_argument("--data_reduction", required = False, default="binning", help='MVA data reduction: "binning" or "peak_picking"')    
    parser.add_argument("--MVA_bin_tof", required = False, default=5, help='if binning: bin tof channels')    
    parser.add_argument("--MVA_bin_pixels", required = False, default=3, help='if binning: bin pixels in x & y direction')    
    parser.add_argument("--MVA_bin_scans", required = False, default=5, help='if binning: bin frames (scans)')    
    parser.add_argument("--min_count", required = False, default=2, help='remove bins/peaks with fewer counts')    
    
    #MVA parameters
    parser.add_argument("--MVA_dimensions", required = False, default=3, help="single or list of dimensions (1,2,3)")    
    parser.add_argument("--MVA_methods", required = False, default="NMF", help='single or list of methods (NMF, PCA)')    
    parser.add_argument("--n_components", required = False, default=5, help='number of components')    
    parser.add_argument("--MVA_scaling", required = False, default="MinMax", help='Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"')    
    
    parser.add_argument("--Top_quantile", required = False, default=False, help='only retain top abundance quantile for MVA')        
    parser.add_argument("--Remove_zeros", required = False, default=True, help='remove pure zero rows and columns')    
    parser.add_argument("--PCA_algorithm", required = False, default="arpack", help='PCA algorithm')    
    parser.add_argument("--NMF_algorithm", required = False, default="nndsvd", help='NMF algorithm')    
    parser.add_argument("--Varimax", required = False, default=False, help='Perform Varimax on PCA loadings')    


    args = parser.parse_args()
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    
    print("")
    print(args) 
    print("")
    locals().update(args)
    


if type(MVA_dimensions)==int or type(MVA_dimensions)==float: MVA_dimensions=[int(MVA_dimensions)]
if type(MVA_methods)==str: MVA_methods=[i.strip() for i in MVA_methods.split(",")]
MVA_dimensions.sort()

fs=str(Path(Output_folder,Path(itmfile).stem)) #base filename for outputs
if not os.path.exists(Output_folder): os.makedirs(Output_folder)
#%%

#get element masses
ifile=str(Path(basedir,"natural_isotope_abundances.tsv"))
emass=0.000548579909 #mass electron
elements=pd.read_csv(ifile,sep="\t") #Nist isotopic abundances and monisiotopic masses
elements=elements[elements["Standard Isotope"]].set_index("symbol")["Relative Atomic Mass"]
elements=pd.concat([elements,pd.Series([-emass,emass],index=["+","-"])]) #add charges

#base calibrants
Calibrants=[
    
            #### positive calibrants ####
            
            
            #common organic
            "CH2+","CH3+","C2H3+","C3H5+","C4H7+","C6H5+","C4H7O+","C5H11+","C8H19PNO4+","C27H45+","C27H45O+","C29H50O2+", 

            #common salts
            "Na+","K+","Ca+","Mg+","Al+","Si+","NaCl+","KNa+","Na_2O+","Na_2OH+","NaO_3H+","KNaOH+","Na_2Cl+","K_2NaSO_4+",
            
            #substrate dependant calibrants
            "Au+","Au_3+"


            #### negative calibrants ####

            #common organic
            "C2-","C3-","CH-","CH2-","C2H4-","C4H-","C_2H_3SO_2-","C16H31O2−","C18H33O2−","C18H35O2−","C27H45O−","C29H49O2−"                           #negative calibrants
            #,"S-","O2-",
            
            #common salts
            "Cl-","SO-","NaCl-","SO_2-","SO_3-","NaClCl-","SO_4H-","NaSO_4-",
            
            #substrate dependant calibrants
            "Au-","Au2-","Au3-"
            
            ]


#%% Functions

def m2c(m,sf,k0):    return np.round(  ( sf*np.sqrt(m) + k0 )  ).astype(int)
def m2cf(m,sf,k0):   return  ( sf*np.sqrt(m) + k0 )   #float version
def c2m(c,sf,k0):    return ((c - k0) / (sf)) ** 2   
def residual(p,x,y): return (y-c2m(x,*p))/y
def centroid(x, y):  return np.sum(x * y) / np.sum(y)

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
    


def CalibrateGlobal(channels,calibrants,sf,k0,
                    ppm_cal=2000,
                    min_mass=10,
                    plot=True):

   
        try: 
            #%%
            # #test
            # sf,k0=I.sf,I.k0
            # channels=centroids
            # ppm_cal=2000
            # min_mass=10
            # plot=True


            #get calibrants
            fcalibrants=calibrants[calibrants>min_mass]
            channels=np.sort(channels)
            mp=c2m(channels,sf,k0)
            q1=find_closest(mp,fcalibrants)
            mp=mp[q1]
        
            ppms=abs((fcalibrants-mp.reshape(-1,1))/mp.reshape(-1,1))*1e6
            q=np.argwhere(ppms<=ppm_cal)
            qmp=mp[q[:,0]]
            qcal=fcalibrants[q[:,1]]
            ppm=(qmp-qcal)/qcal*1e6
            
            #1st calibration    
            x,y=channels[q1[q[:,0]]],qcal
            
            #filter ppm outliers
            pm=ppm-np.median(ppm)
            q25,q75=np.percentile(pm,25),np.percentile(pm,75) 
            q2=(pm<q75+1.5*(q75-q25)) & (pm>q75-1.5*(q75-q25))
            
            x,y=x[q2],y[q2]
            

      
            [sf,k0], _ = curve_fit(c2m,x,y,p0=[sf,k0])
            post_ppms=(c2m(x,sf,k0)-y)/y*1e6
           
            #filter calibrants
            pm=post_ppms-np.median(post_ppms)
            q1,q3=np.percentile(pm,25),np.percentile(pm,75) 
            q=(pm<q3+1.5*(q3-q1)) & (pm>q3-1.5*(q3-q1))
            
            #2nd calibration
            [sf,k0], _ = curve_fit(c2m,x[q],y[q],p0=[sf,k0])
            post_ppms2=(c2m(x[q],sf,k0)-y[q])/y[q]*1e6
            
            if plot:
                
                pret=str(round(sum(abs(ppm[q2][q]))/len(ppm[q2][q]),1))
                postt=str(round(sum(abs(post_ppms2))/len(post_ppms2),1))
    
                #plotting
                fig,ax=plt.subplots()
                plt.scatter(y[q],ppm[q2][q],c=[(0.5, 0, 0, 0.3)])
                plt.scatter(y[q],post_ppms2,c=[(0, 0.5, 0, 0.3)])
                plt.legend(["pre calibration","post calibration"])
                plt.xlabel("m/z")
                plt.ylabel("ppm mass error")
                plt.title("global_calibration")
                fig.savefig(fs+"_glob_cal_scat.png",bbox_inches="tight",dpi=300)
             
                fig,ax=plt.subplots()
                y1, _, _ =plt.hist(ppm[q2][q],color=(0.5, 0, 0, 0.3))
                y2, _, _ =plt.hist(post_ppms2,color=(0, 0.5, 0, 0.3))
                plt.vlines(np.mean(ppm[q2][q]),0,np.hstack([y1,y2]).max(),color=(0.5, 0, 0, 1),linestyle='dashed')
                plt.vlines(np.mean(post_ppms2),0,np.hstack([y1,y2]).max(),color=(0, 0.5, 0, 1),linestyle='dashed')
                plt.xlabel("ppm mass error")
                plt.ylabel("frequency")
                plt.legend(["pre: mean "+str(round(np.mean(ppm),1))+ ", abs "+pret,
                            "post: mean "+str(round(np.mean(post_ppms2),1))+ ", abs "+postt],
                            loc=[1.01,0])
                plt.title("global_calibration")
                fig.savefig(fs+"_glob_cal_hist.png",bbox_inches="tight",dpi=300)
                
                
#%%
        except:
            pass
            print("calibration failed!")
     
        pd.DataFrame([[sf,k0]],columns=["sf","k0"]).to_csv(fs+"calib.csv")

        return sf,k0


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
                
                        if plot_non_selected:
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

        if plot_selected:
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

#%%









I = pySPM.ITM(itmfile)

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

### Get ITM metadata ###
xpix,ypix= math.ceil(I.size["pixels"]["x"]), math.ceil(I.size["pixels"]["y"])
scans=math.ceil(I.Nscan)
k0,sf=I.k0,I.sf
mode_sign={"positive":"+","negative":"-"}.get((I.polarity).lower())
calibrants=np.sort(np.array([pySPM.utils.get_mass(i) for i in Calibrants if i.endswith(mode_sign)])) #shortended calibrants list
x_um,y_um=I.size["real"]["x"]*1e6,I.size["real"]["y"]*1e6
sputtertime=I.get_value("Measurement.SputterTime")["float"] #not sure if correct


#%% To Do


#use user supplied ROIs?
#add optional calibrant txtfile?
#ROI plot: check if rotation is correct

sSpectrum,ps,pns=pick_peaks(ds)  



###### Global calibration ######

sSpectrum,ps,pns=pick_peaks(ds)                                                                    #this only picks "single peaks"
centroids=np.array([centroid(sSpectrum[i[1]:i[2]].index,sSpectrum[i[1]:i[2]].values) for i in ps]) #calculate peak centroids
sf,k0,=CalibrateGlobal(centroids,calibrants,sf,k0,ppm_cal=2000)                                     #calibrate

np.save(fs+"_summed_spectrum",sSpectrum) #save summed spectrum

######  Get mass resolution ###### 

#linear fit of peak resolution
x,y=ps[:,0],(ps[:,2]-ps[:,1])#/2
s=np.argsort(x)
x,y=x[s],y[s]    
A = np.vstack([x, np.ones(len(x))]).T
[a, b], r = np.linalg.lstsq(A, y)[:2]
r2 = 1 - r / (y.size * y.var())

#plot resolution fit
fig,ax=plt.subplots()
plt.scatter(x,y)
plt.plot(x,x*a+b,color="red")
plt.xlabel("mass channel")
plt.ylabel("fwhm in channels")
plt.legend(["single peaks","linear fit, r2: "+str(np.round(r2,3)[0])])
fig.savefig(fs+"_channel_res.png",bbox_inches="tight",dpi=300)
            
###### Filter channels on peakwidth ###### 
   
p=np.vstack([ps,pns])
pw=p[:,2]-p[:,1]
rs=pw/(p[:,0]*a+b)
q=(rs>=min_width_ratio) & (rs<=max_width_ratio)
lb,rb=p[q,1].astype(int),p[q,2].astype(int)

fig,ax=plt.subplots()
plt.hist(rs,bins=50)
plt.vlines([min_width_ratio,max_width_ratio],0,ax.get_yticks()[-1],color="r",linestyle="--")
plt.title("peak width distribution")
plt.xlabel("ratio to expeted peakwidth")
plt.legend(["width cutoffs","peak width ratio"])
fig.savefig(fs+"_peak_widths.png",bbox_inches="tight",dpi=300)

ds=ds[np.in1d(ds.tof,create_ranges(np.vstack([lb,rb]).T))]

###### Assign peaks ######

pmat=np.zeros(p.max()+1,dtype=np.int64)
pmat[create_ranges(p[:,1:])]=np.repeat(np.arange(len(p)),p[:,2]-p[:,1])
ds["peak_bin"]=p[pmat[ds["tof"]],0]




#%%

###### ROI detection #####
if ROI_dimensions<=1 or ROI_clusters<=1:
    ds["ROI"]=0
else:

    if ROI_dimensions==2: gcols=["x","y"]
    if ROI_dimensions==3: gcols=["x","y","scan"]
    col="peak_bin"
    
    #binning
    rds=ds.copy()
    if ROI_bin_pixels: rds[["x","y"]]=(rds[["x","y"]]/ROI_bin_pixels).astype(int)
    if ROI_bin_scans: rds["scan"]=(rds["scan"]/ROI_bin_scans).astype(int) 
    
    cd=rds.groupby(gcols+[col]).size().to_frame("count").reset_index()
    
    #limit to x top peaks
    cds=cd.groupby("peak_bin")["count"].sum()
    if len(cds)>ROI_peaks: cd=cd[cd["peak_bin"].isin(cds.sort_values().index[:ROI_peaks])] 
    
    #construct peak matrix
    ut=np.unique(cd[col])
    uz=np.zeros(cd[col].max()+1,dtype=np.uint32)
    uz[ut]=np.arange(len(ut))
    xys=cd[gcols].values
    split_at=np.argwhere(np.any(np.diff(xys,axis=0)!=0,axis=1))[:,0]
    sxys=np.vstack([xys[split_at],xys[-1]])
    sx=np.array_split(cd[[col,"count"]].values.astype(np.uint32),split_at+1)
    szm=np.zeros([len(sxys),len(ut)],dtype=bits(cd["count"].max()))  
    for ix,i in enumerate(sx): szm[ix,uz[i[:,0]]]=i[:,1]
    
    #scaling
    if ROI_scaling=="Robust":   szm=robust_scale(szm,axis=0)
    if ROI_scaling=="Standard": szm=(szm-szm.mean(axis=0))/szm.std(axis=0) #z = (x - u) / s
    if ROI_scaling=="Poisson":  szm=szm/np.sqrt(szm.mean(axis=0))  #2D poisson scaling X/ sqrt mean  /np.sqrt(szm.mean(axis=1))/
    if ROI_scaling=="MinMax":   
        mins=(szm.max(axis=0)-szm.min(axis=0))
        mins[mins==0]=1 #avoid nan
        szm=(szm-szm.min(axis=0))/mins
    if ROI_scaling=="Jaccard":  szm=szm.astype(bool).astype(int)
    szm[np.isnan(szm)]=0 #this doesnt work
    szm=csr_matrix(szm).T
    
    #Kmeans
    kmeans = KMeans(n_clusters=ROI_clusters, random_state=0, n_init="auto").fit(szm.T)
    rois=kmeans.labels_
    
    #map to space for indexing
    if ROI_dimensions==2: 
        zroi=np.zeros([math.ceil(xpix/ROI_bin_pixels),math.ceil(ypix/ROI_bin_pixels)],dtype=np.int8)
        zroi[sxys[:,0],sxys[:,1]]=rois
    if ROI_dimensions==3: 
        zroi=np.zeros([math.ceil(xpix/ROI_bin_pixels),math.ceil(ypix/ROI_bin_pixels),math.ceil(scans/ROI_bin_scans)],dtype=np.int8)
        zroi[sxys[:,0],sxys[:,1],sxys[:,2]]=rois
 
    
    
    np.save(fs+"_ROImap",zroi) #save ROI map
    
    #plot 2D ROIs
    if ROI_dimensions==2:
        fig,ax=plt.subplots()
        g=sns.heatmap(rois.reshape(math.ceil(xpix/ROI_bin_pixels),math.ceil(ypix/ROI_bin_pixels)).T)
        
        #update to pixels to micrometers
        ax.set_yticklabels((np.linspace(0,y_um,len(g.get_yticks()))).astype(int),rotation=0)
        ax.set_ylabel(r'y $\mu$m') 
        
        ax.set_xticklabels((np.linspace(0,x_um,len(g.get_xticks()))).astype(int),rotation=90)
        ax.set_xlabel(r'x $\mu$m')
        
        fig.savefig(fs+"_ROI_2D.png",dpi=300) #save 2D ROI plot

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
                      color='r', size_max=18,opacity=0.02) 
        
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
        
        fig.show()
        fig.write_html(fs+"_ROI_3D_combined.html") #save 3d ROI plot
    
        #plot separate segments
        #%%
        roi_reorder=rdf["r"].drop_duplicates().values.astype(int)
        for ix_roi,roi in enumerate(roi_reorder):
            
            d=rdf[rdf["r"]==str(roi)]
            fig = go.Figure(data=[go.Scatter3d(x=d["y"], y=d["x"], z=d["z"],
                                   mode='markers',opacity=0.02)])
            
            
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
    

    
            fig.update_layout(scene_camera=camera, title="ROI 3D segment "+str(roi))
            
            fig.show()
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


ss=[]
for roi in range(ROI_clusters): #Depth profile per ROI
    
    rcd=cd[cd["ROI"]==roi]
    
    if normalize: #normalize to total count per scan
        rcd["count"]=rcd["count"]/rcd.groupby("scan")["count"].sum().loc[rcd.scan].values 
        
    if smoothing: #smooth with rolling mean
        scan_arr=np.arange(scans)
        missing_scans=[] #add missing scans
        for n,g in rcd.groupby("peak_bin",sort=False)["scan"]:
            if len(g)!=len(scan_arr):
                ms=scan_arr[~np.in1d(scan_arr,g)]
                missing_scans.append(np.vstack([np.repeat(n,len(ms)),ms]).T)
        rcd=pd.concat([rcd,pd.DataFrame(np.vstack(missing_scans),columns=["peak_bin","scan"])]).fillna(0).astype(int).sort_values(by=["peak_bin","scan"]).reset_index(drop=True)
        rcd=rcd.groupby("peak_bin").rolling(smoothing).mean().reset_index()[['peak_bin', 'scan', 'count']].fillna(0) 
    
    rcd["peak_bin"]=c2m(rcd["peak_bin"],sf,k0)
    
    if ROI_clusters>1: rcd.to_csv(fs+"_ROI"+str(roi)+"_depth_profile.tsv",sep="\t")  #save depth profile
    else:              rcd.to_csv(fs+"_depth_profile.tsv",sep="\t")  #save depth profile
    
    ss.append(rcd.groupby("peak_bin")["count"].sum())

##### Depth profile Biplot ######

#https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot
#https://statomics.github.io/HDDA/svd.html#7_SVD_and_Multi-Dimensional_Scaling_(MDS)   
#https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html


ss=pd.concat(ss,axis=1).sort_index()
ss.columns=[np.arange(ROI_clusters)]
ss.index=np.round(ss.index,2)

#principle component analysis
ss=(ss-ss.min())/(ss.max()-ss.min()) #minmax scaling
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


###### Dimension reduction ######

ds=ds[c2m(ds["tof"],sf,k0)<max_mass] #truncate masses

if data_reduction=="peak_picking": 
    bin_tof,bin_pixels,bin_scans=1,1,1 #turn off binning

#binning
ds["tof"]=  (ds["tof"] /MVA_bin_tof   ).astype(int) 
ds[["x","y"]]=(ds[["x","y"]]/MVA_bin_pixels).astype(int)
ds["scan"]=  (ds["scan"] /MVA_bin_scans   ).astype(int) 
ds=ds.astype(np.uint32)
xpix,ypix,scans= xpix/MVA_bin_pixels,ypix/MVA_bin_pixels,scans/MVA_bin_scans


###### Create peak matrix #####
for MVA_dimension in MVA_dimensions:
    
    
    if MVA_dimension==1: gcols=["scan","ROI"]
    if MVA_dimension==2: gcols=["x","y"]
    if MVA_dimension==3: gcols=["x","y","scan"]
    
    #turn off ROI for dimensions>1
    if MVA_dimension>1: 
        ROI_clusters=0
        ds["ROI"]=0 
    
    
    if data_reduction=="peak_picking": col="peak_bin"
    if data_reduction=="binning":      col="tof"
    
    
    
    for roi in range(ROI_clusters+1):
        
        cd=ds[ds["ROI"]==roi].groupby(gcols+[col]).size().to_frame("count").reset_index()
        
        if len(cd):
            #min count filtering
            q75=np.quantile(cd["count"],0.75)
            if q75>min_count: cd=cd[cd["count"]>=min_count]
            else: print("Minimum count too high, skipping filtering!")        
                
            
                
            
            ut=np.unique(cd[col])
            uz=np.zeros(cd[col].max()+1,dtype=np.uint32)
            uz[ut]=np.arange(len(ut))
            
            xys=cd[gcols].values
            split_at=np.argwhere(np.any(np.diff(xys,axis=0)!=0,axis=1))[:,0]
            
            sxys=np.vstack([xys[split_at],xys[-1]])
            sx=np.array_split(cd[[col,"count"]].values.astype(np.uint32),split_at+1)
            szm=np.zeros([len(sxys),len(ut)],dtype=bits(cd["count"].max()))  
            for ix,i in enumerate(sx): szm[ix,uz[i[:,0]]]=i[:,1]
                
            if  Top_quantile: #only keep top abundant  
                szm[:,np.argwhere(szm.mean(axis=0)<np.quantile(szm.mean(axis=0),Top_quantile))]=0 
            
            if Remove_zeros: 
                q=szm.sum(axis=0)>0 
                ut=ut[q]
                szm=szm[:,q]
               
            if szm.shape[0]<n_components: n_components=szm.shape[0] 
               
            ### MVA Scaling 
            if MVA_scaling=="Standard": szm=(szm-szm.mean(axis=0))/szm.std(axis=0) #z = (x - u) / s
            if MVA_scaling=="Robust":   szm=robust_scale(szm,axis=0)
            if MVA_scaling=="Poisson":  szm=szm/np.sqrt(szm.mean(axis=0))  #2D poisson scaling X/ sqrt mean  /np.sqrt(szm.mean(axis=1))
            if MVA_scaling=="MinMax":   
                mins=(szm.max(axis=0)-szm.min(axis=0))
                mins[mins==0]=1 #avoid nan
                szm=(szm-szm.min(axis=0))/mins
            if MVA_scaling=="Jaccard":  szm=szm.astype(bool).astype(int)
            
            szm=csr_matrix(szm).T
        
            
            for MVA_method in MVA_methods:
                   
                if MVA_method=="NMF":
                    model = NMF(n_components=n_components, init=NMF_algorithm, random_state=0, verbose=True,max_iter=20000)
                    MVA = model.fit_transform(szm)
                    loadings=model.components_
                    
                if MVA_method=="PCA":
                    clf = TruncatedSVD(n_components=n_components,algorithm=PCA_algorithm) #random is fast, arpack is more accurate
                    MVA = clf.fit_transform(szm.astype(float))
                    if Varimax: MVA =varimax(MVA)
                    loadings=clf.components_* np.sqrt(clf.explained_variance_).reshape(-1,1)
            
        
                ldf=pd.DataFrame(np.hstack([sxys,loadings.T]),columns=gcols+np.arange(n_components).tolist())
                                
                if data_reduction=="binning": #group by peak
                    mdf=pd.DataFrame(np.hstack([c2m(ut*MVA_bin_tof,sf,k0).reshape(-1,1),MVA]),columns=["mass"]+np.arange(n_components).tolist())
                    mdf["peak_mass"]=c2m(p[pmat[m2c(mdf.mass,sf,k0)],0],sf,k0)
                    mdf=mdf.groupby("peak_mass",sort=False)[np.arange(n_components).tolist()].sum()
                else:
                    mdf=pd.DataFrame(np.hstack([c2m(ut,sf,k0).reshape(-1,1),MVA]),columns=["peak_mass"]+np.arange(n_components).tolist())
                   
                
                if ROI_clusters>1: #save with ROI
                    ldf.to_csv(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_loadings.csv")
                    mdf.to_csv(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.csv") 
                else:
                    ldf.to_csv(fs+"_"+str(MVA_dimension)+"D_"+MVA_method+"_loadings.csv")
                    mdf.to_csv(fs+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.csv")
                
                  #%%
                    
                ### plot mass loadings
                rows,cols=n_components//5,5
                if n_components%5: rows+=1
                if cols>n_components: cols=n_components
                
                fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                axes = axes.flatten()
                for i in range(n_components):
                    axes[i].plot(c2m(ut*MVA_bin_tof,sf,k0),MVA[:,i])
                
                for ix,i in enumerate(axes):
                    i.title.set_text(ix+1)
                
                if ROI_clusters>1: fig.savefig(fs+"_ROI"+str(roi)+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.png",dpi=300) #save summed intensity png
                else:              fig.savefig(fs+"_"+str(MVA_dimension)+"D_"+MVA_method+"_components.png",dpi=300) #save summed intensity png     
                    
                ###### Spatial plots
                
            
          
                #%% Plot 1D
                
                if MVA_dimension==1:
                    
                    rows,cols=n_components//5,5
                    if n_components%5: rows+=1
                    if cols>n_components: cols=n_components
                    
                    fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                    axes = axes.flatten()
                    for i in range(n_components):
                        
                        axes[i].plot(sxys*sputtertime/scans,loadings[i,:])
                        
                        
                    for ix,i in enumerate(axes):
                        i.title.set_text(ix+1)
                        
                    fig.supylabel('Loading')
                    fig.supxlabel('Sputter time [s]')
                    
                    if ROI_clusters>1: fig.savefig(fs+"_ROI"+str(roi)+"_"+MVA_method+"_1D_loadings.png",dpi=300) #save summed intensity png
                    else:              fig.savefig(fs+"_"+MVA_method+"_1D_loadings.png",dpi=300) #save summed intensity png
                    
                    #pils save plot (add ROI incase ROI clusters)
                
                #%% Plot 2D
                
                if MVA_dimension==2:
                
                    rows,cols=n_components//5,5
                    if n_components%5: rows+=1
                    if cols>n_components: cols=n_components
                    
                    fig,axes=plt.subplots(rows,cols,figsize=(cols*4,rows*4))
                    axes = axes.flatten()
                    for i in range(n_components):
                        
                        hm=pd.DataFrame(np.hstack([sxys,loadings[i,:].reshape(-1,1)]),columns=["x","y","c"]).pivot(columns="x",index="y").fillna(0)
                        hm.columns = hm.columns.droplevel().astype(int)
                        hm.index=hm.index.astype(int)
                        
                        g=sns.heatmap(hm,ax=axes[i],cbar=False,center=0,robust=True)
                        
                        #update to pixels to micrometers
                        axes[i].set_yticklabels((np.linspace(0,y_um,len(g.get_yticks()))).astype(int),rotation=0)
                        if ix: axes[i].set_ylabel(r'y $\mu$m') 
                        else:  axes[i].set_ylabel('') #prevent bleed
                        
                        axes[i].set_xticklabels((np.linspace(0,x_um,len(g.get_xticks()))).astype(int),rotation=90)
                        axes[i].set_xlabel(r'x $\mu$m')
                        
                    for ix,i in enumerate(axes):
                        i.title.set_text(ix+1)
                    
                    fig.savefig(fs+"_"+MVA_method+"_2D_loadings.png",dpi=300) #save summed intensity png
                
                
                #%% Plot 3D
                
                if MVA_dimension==3:
                
     
                    cmax=max(loadings.max(),abs(loadings.min()))
                    #### Surface plot
                    
                    for i in range(n_components):
                   
                        vdf=pd.DataFrame(np.vstack([sxys.T,loadings[i,:].reshape(1,-1)]).T,columns=["x","y","scan","counts"])
                    
                        X,Y,Z=np.meshgrid(np.arange(math.ceil(xpix)), np.arange(math.ceil(ypix)),np.arange(math.ceil(scans)))
                        vm=np.zeros([math.ceil(xpix),math.ceil(ypix),math.ceil(scans)])
                        vm[vdf["x"].astype(int).tolist(),vdf["y"].astype(int).tolist(),vdf["scan"].astype(int).tolist()]=vdf["counts"].values
                
                        
                        x,y,z,v=X.flatten(),Y.flatten(),Z.flatten(),vm.flatten()
                        
                        #remove background points
                        q=np.argwhere(abs(v)>=(np.quantile(abs(v),0.90)/10))
          
                        
                        
                        
                        x,y=x*x_um/xpix,y*y_um/ypix #convert to micrometer
                        z=z*sputtertime/scans       #convert to sputter time in seconds
                       
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
                
                
                        fig.update_layout(scene_camera=camera, title="Component: "+str(i+1))
                        
                        fig.show()
                        fig.write_html(fs+"_"+MVA_method+"_3D_comp"+str(i)+"loading.html") #save summed intensity 
                
                
    

