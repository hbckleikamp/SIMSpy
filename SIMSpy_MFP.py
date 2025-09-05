# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:10:11 2025

@author: e_kle
"""


import os 
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import json

#for ppm fitting
from skmisc import loess
from moepy import lowess

#my own modules
import cart2form
import HorIson
# %% change directory to script directory (should work on windows and mac)
from inspect import getsourcefile
basedir = str(Path(os.path.abspath(getsourcefile(lambda: 0))).parents[0])
os.chdir(basedir)
print(os.getcwd())
base_vars=list(locals().copy().keys()) #base variables

#%% Parameters


### Input ###
input_files=[]   #required
MFP_db=""        #required
input_type="ROI" #ROI or MFP
Output_folder=""

### MFP ###
ppm=500
top_candidates=100  #500

### recalibration ###
recalibrate=True 
post_calib_ppm=200
Substrate="Au"
Substrate_Calibrants=str(Path(basedir,"Substrate_Calibrants.csv"))  #list of ions coming from substrate
Calibrants=str(Path(basedir,"Calibrants.csv"))                      #list of typical background ions

### Isotope filtering ###
isotope_range=[-2,6]
min_intensity=10    

write_params=True
load_params=""

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
    parser.add_argument("-i", "--input_folders", required = True,  help="Required: putput folders of SIMSpy MVA")
    parser.add_argument("-d","--MFP_db", required = True, default="", help="Molecular formula database, constructed with space2cart.py")
    parser.add_argument("-o", "--Output_folder", required = False, default="", help="Output folder")

    #MFP
    parser.add_argument("--ppm", required = False, default=500, help="Maximum ppm mass error of initial MFP")
    parser.add_argument("--top_candidates", required = False, default=100, help="number of top candidates in output (sorted by ppm)")

    #Recalibrate
    parser.add_argument("--recalibrate", required = False, default=True, help="Perform mass recalibration after 1st MFP")
    parser.add_argument("--post_calib_ppm", required = False, default=200, help="Maximum ppm mass error for MFP after recalibration")
    parser.add_argument("--Substrate", required = False, default="", help="Add Substrate calibrants that contain these elements")
    parser.add_argument("--Substrate_Calibrants", required = False, default=str(Path(basedir,"Substrate_Calibrants.csv")) , help="List of Substrate Calibrants or path to file")
    parser.add_argument("--Calibrants", required = False, default=str(Path(basedir,"Calibrants.csv"))   , help="List of Sample Calibrants or path to file")
    
    #Isotope filtering
    parser.add_argument("--isotope_range", required = False, default=[-2,6] , help=' minimum and maximum isotope considered')
    parser.add_argument("--min_intensity", required = False, default=10 , help=' minimum counts for simulated isotopes')


    parser.add_argument("--write_params",  required = False, default=True, help="write parameters used to file")
    parser.add_argument("--load_params",  required = False, default="", help="use params from a file")
    
    


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


if type(input_files)==str:  input_files=input_files.split(",")


isotope_range=np.arange(isotope_range[0],isotope_range[1]+1)

#%%



#### Isotopic data
mass_table = str(Path(basedir, "isotope_table.tsv"))           # table containing element masses, default: CartMFP folder/ mass_table.tsv"
if os.path.exists(mass_table):
    tables = pd.read_csv(mass_table, index_col=[0], sep="\t")


emass = 0.000548579909  # electron mass
element_masses=tables[tables["delta neutrons"]==0].set_index("symbol")['Relative Atomic Mass']
element_masses.loc["+"]=-emass
element_masses.loc["-"]=emass






# #https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
def mc(x, y):
    return np.einsum('ij,ij->i', x, y) / (
              np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )

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

#https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
def create_ranges(a):
    l = a[:,1] - a[:,0]
    clens = l.cumsum()
    ids = np.ones(clens[-1],dtype=int)
    ids[0] = a[0,0]
    ids[clens[:-1]] = a[1:,0] - a[:-1,1]+1
    return ids.cumsum()




#parse form
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
    return (cdf.values*element_masses.loc[cdf.columns].values).sum() / cdf[["+","-"]].sum(axis=1)

def find_background(Calibrants,masses):

    bions=pd.concat([parse_form(i) for i in Calibrants]).fillna(0).astype(int)
    bions=bions.loc[:,bions.max()>0]
    hill=bions.columns.sort_values().tolist()
    hill=hill[2:]+hill[:2]

    #construct element string
    ecounts=bions[hill]
    e_arr=np.tile(hill,len(bions)).reshape(len(bions),-1) 
    e_arr=np.where(ecounts==0,"",e_arr)
    eles=ecounts.applymap(str).replace("0","").replace("1","")
    bions.index=["".join(i) for i in np.hstack([e_arr,eles])[:,np.repeat(np.arange(len(hill)),2)+np.tile(np.array([0,len(hill)]),len(hill))]]
    
    bions["mass"]=np.sum(bions.values*element_masses.loc[bions.columns].values,axis=1)
    bions=bions.sort_values(by="mass")

    tree = KDTree(masses.values.reshape(1,-1).T, leaf_size=200) 
    
    t=tree.query_radius(bions.mass.values.reshape(1,-1).T,r=bions.mass.values*ppm/1e6)
    tdf=pd.DataFrame(np.hstack(t),index=np.repeat(bions.index,[len(i) for i in t]),columns=["ix"])
    bions=bions.loc[tdf.index]
    tdf["original_index"]=monos.iloc[tdf["ix"]].index
    tdf.pop("ix")
    
    #make same format as mfp output
    tdf["index"]=tdf["original_index"]
    if mode=="neg": tdf[['adduct','adduct_mass']]=["+-",emass]
    if mode=="pos": tdf[['adduct','adduct_mass']]=["--",-emass]
    tdf["charge"]=1 #charge is not really used atm
    tdf["rdbe"]=1
    tdf["rdbe"]+=bions.loc[:,bions.columns.isin(["C"])].sum(axis=1).values
    tdf["rdbe"]-=bions.loc[:,bions.columns.isin(["H","F","Cl","Br","I"])].sum(axis=1).values/2
    tdf["rdbe"]+=bions.loc[:,bions.columns.isin(["N","P"])].sum(axis=1).values/2
    
    tdf["input_mass"]=monos.loc[tdf["original_index"]].mass.values
    tdf["mass"]=tdf["input_mass"]-tdf["adduct_mass"]
    tdf["pred_mass"]=bions.mass
    tdf["ppm"]=(tdf["pred_mass"]-tdf["input_mass"])/tdf["pred_mass"]*1e6 #calculate ppm
    tdf["appm"]=tdf["ppm"].abs()
   
    c=sorted(set(bions.columns[:-1])-set(["+","-"]))
    tdf[c]=bions[c]
    
    tdf["formula"]=tdf.index.str.strip("+-")
    tdf["formula+adduct"]=tdf.index.str.strip("+-")
    tdf=tdf[tdf["appm"]<=ppm]
    #%%
    return tdf,c
        



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



def MFP(masses,ppm):
    
    
    mfp=cart2form.predict_formula(masses,composition_file=MFP_db, #MFP_dbunfilt_path,
                                  ppm=ppm,top_candidates=top_candidates,
                                  mode=mode,adducts=adducts,keep_all=False,add_formula=True)

    #annotate background ions and add to mfp
    if len(Calibrants):
        
        tdf,c=find_background(Calibrants,masses)
        
        #add background and reorder columns
        cq=mfp.columns.isin(element_masses.index)
        me=mfp.columns[cq].tolist()
        mne=mfp.columns[~cq][:-2].tolist()
        mfp=pd.concat([mfp,tdf])
        mfp=mfp[mne+me+sorted(set(c)-set(me))+["formula","formula+adduct"]].fillna(0).reset_index(drop=True)
        
    return mfp

    

#%% 



filenames=[]
dfs=[]

for ifile in input_files:
    
    if input_type=="ROI": 
        df=pd.read_csv(ifile,sep="\t")
        dfs.append(df)
        filenames.append(ifile)
        
        
    if input_type=="MVA": 
        
        df=pd.read_csv(ifile)
        for i in df.columns[1:].tolist():
            x=df[["peak_mass",i]]
            x.columns=["mass","Apex"]
            dfs.append(x)
            filenames.append(ifile.replace("components.csv","component_"+str(i)+".csv"))


    

    
    
    
    
    
#%%



#%% make masses unique
for file_ix,ifile in enumerate(filenames):
    


    fs=Path(ifile).stem
    if os.path.exists(Output_folder): fss=str(Path(Output_folder,fs))
    else:                             fss=str(Path(Path(ifile).parents[0],fs))                          
    
    lrecalibrate=recalibrate #reset in each loop

    #read isotopic information
    if   "(-)" in ifile:  mode,mode_sign,adducts="neg","-",["-H+",'+-'] #test ['+-']
    elif "(+)" in ifile:  mode,mode_sign,adducts="pos","+",["+H+",'--']  #test ['--']

    
    df=dfs[file_ix]
    if "index" not in df.columns: df["index"]=np.arange(len(df))
    df=df.set_index("index")
    if not "isotope" in df.columns: df["isotope"]=0
    monos=df[df["isotope"]==0]
    
    #Non 7gr annotation
    mfp=MFP(monos.mass,ppm=ppm)
    if not len(mfp):
        print("no formulas found")
        continue

    

    #make unique formulas
    mfp=mfp.sort_values(by=["index","formula+adduct","appm"])



    ## Recalibration

    if lrecalibrate:

#%%
        
        calibrants=[i[:-1] for i in Calibrants if i[-1]==mode_sign]
        u=mfp[mfp["formula+adduct"].isin(calibrants)]
        u["intensity"]=monos.loc[u["original_index"],"Apex"].values
        u=u.sort_values(by=["formula+adduct","intensity"],ascending=False)
        u=u.groupby("formula+adduct",sort=False).nth(0).sort_values(by="input_mass")
        p=u[["input_mass","ppm"]]
        
        if len(p)>3:
        
            #quantile denoising nearest neighbours
            tree = KDTree(p.input_mass.values.reshape(1,-1).T, leaf_size=20) 
            _,t=tree.query(p.input_mass.values.reshape(1,-1).T,k=min(len(p),5))
            cd=np.mean(abs(p.ppm.values[t]-p.ppm.values.reshape(-1,1)),axis=1)
            med=np.median(cd)
            qmin=med-(med-np.quantile(cd,0.25))*1.5
            qmax=med+(np.quantile(cd,0.75)-med)*1.5
    
            q=(cd>qmin) & (cd<qmax)
            p=p[q]
            pr=p.rolling(window=3).mean().dropna()
            xf,yp=pr.input_mass.values,pr.ppm.values
      
            df.mass=df.mass*(1-np.interp(df["mass"].values.astype(np.float64),xf,yp)/1e6)
            monos=monos=df[df["isotope"]==0]
            ppm=post_calib_ppm
            mfp=MFP(monos.mass,ppm=ppm)
    
        
            
            if len(xf):
                fig,ax=plt.subplots()
                plt.scatter(p.input_mass,p.ppm,label="pre calibration",s=10)
                plt.scatter(p.input_mass,p.ppm-np.interp(p.input_mass,xf,yp),s=10,label="post calibration")
                
                plt.legend()
                plt.xlabel("mz")
                plt.ylabel("ppm")
                plt.title(fss)
                fig.savefig(fs+"_calibration.png",bbox_inches="tight",dpi=300)
    

       
            
     
    mfp=mfp.sort_values(by=["input_mass","formula+adduct","appm"]).reset_index(drop=True)
    mfp["Mono_intensity"]=monos.Apex.loc[mfp["original_index"]].values
    
    elements=mfp.columns[mfp.columns.isin(element_masses.index)]
    q=mfp[elements].max()==0
    [mfp.pop(i) for i in np.array(elements)[q]]
    elements=mfp.columns[mfp.columns.isin(element_masses.index)]
 
    #%% Isotope prediction
   
    

    imass=HorIson.multi_conv(mfp[elements],isotope_range=isotope_range)
    imass["abundance"]/=imass.loc[imass.isotope==0,"abundance"].loc[imass.index] #normalize to mono
    imass["abundance"]*=mfp["Mono_intensity"].loc[imass.index]                   #correct with mono mfp
    imass=imass[imass["abundance"]>=min_intensity]


    dm,im=df.mass.values,imass.mass.values
    
    #correct with charge
    if mode=="neg": dm-=emass
    if mode=="pos": dm+=emass
    
    #correct with ppm
    im*=(1-mfp.loc[imass.index,"ppm"].values/1e6)
    tree = KDTree(im.reshape(1,-1).T, leaf_size=200) 
    t=tree.query_radius(dm.reshape(1,-1).T,r=dm*post_calib_ppm/1e6)
    u=[np.vstack([np.repeat(ix,len(i)),i]).T for ix,i in enumerate(t) if len(i)]
    idf=pd.DataFrame(np.vstack(u),columns=["dx","ix"])
    
    idf["dix"]=df.iloc[idf.dx].index.values    #index of measured row
    idf["iix"]=imass.iloc[idf.ix].index.values #index of simulated row
    idf[["diso","dint","mm"]]=df.iloc[idf.dx][["isotope","Apex","mass"]].values
    idf[["siso","sint","mi"]]=imass.iloc[idf.ix][["isotope","abundance","mass"]].values
    
    #groupby iix, siso, lowest ppm
    idf["ppm"]=(idf["mm"]-idf["mi"])/idf["mm"]*1e6
    idf["appm"]=idf["ppm"].abs()
    
    idf=idf.sort_values(by=["iix","siso","appm"]).groupby(["ix","siso"],sort=False).nth(0)
    idf["cint"]=np.clip(idf["dint"],0,idf["sint"])
    idf["eu"]=abs(idf["cint"]-idf["sint"])
    
    seu=idf.groupby(["iix"])[["cint","eu"]].sum()
    seu["neu"]=1-seu["eu"]/seu["cint"]
    mfp["euclidian_sim"]=seu["neu"]     
   
        
    #add isotopes
    sidf=idf.groupby(["iix","siso"])[["sint","dint"]].sum().reset_index() #sum or pick lowest ppm?
    pvs=sidf.pivot(columns="siso",index="iix",values="sint")
    pvi=sidf.pivot(columns="siso",index="iix",values="dint")
    pcols=["m_"+str(i) for i in pvi.columns]
    scols=["s_"+str(i) for i in pvs.columns]
    mfp[pcols]=pvi     
    mfp[scols]=pvs
    


    ## filter on ppm
    fmfp=mfp.sort_values(by=["original_index","appm"]).groupby("original_index",sort=False).nth(0)
    
    
    q=fmfp["input_mass"].values<200
    x1,y,w=fmfp["input_mass"].values[q],fmfp["ppm"].values[q],fmfp["Mono_intensity"].values[q]
    
    p1=np.convolve(y, np.ones(10), 'valid') / 10 #rolling mean
    try: #weighted lowess
        l=loess.loess(x1, y, weights=np.log2(w),span=0.75)
        l.fit()
        p1 = l.predict(x1).values
    except: #normal lowess
        lm = lowess.Lowess()
        lm.fit(x1,y, frac=.75, num_fits=100) #lowess fit introduces randomness each time
        p1=lm.predict(x1)
    

    x1=x1[:len(p1)]
    y=y[:len(p1)]
    
    fig,ax=plt.subplots()
    plt.scatter(x1, y,s=0.5)
    plt.plot(x1, p1,label="0 - 200")
    plt.title(fs)

    fig.savefig(fs+"_post_mfp_calibration.png",bbox_inches="tight",dpi=300)
    mfp["dppm"]=abs(mfp["ppm"]-np.interp(mfp["input_mass"],x1,p1))
    mfp.to_csv(fss+"_best_mfp.tsv",sep="\t")
    
    #%% add isotope formulas
    
    idf["dppm"]=abs(idf["ppm"]-np.interp(idf["mm"],x1,p1))
    idf["formula"]=mfp["formula+adduct"].loc[idf.iix].values
    q=idf["siso"]!=0
    idf.loc[q,"formula"]+="^"+idf.loc[q,"siso"].astype(float).astype(int).astype(str)
    idf.to_csv(fss+"_peak_mfp.tsv",sep="\t")
