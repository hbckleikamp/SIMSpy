# SIMSpy

SIMSpy is a collection of python routines that enable untargeted multivariate analysis of TOF-SIMS data, molecular formula prediction, and targeted analysis of fragments.
it relies on the module pySPM (https://github.com/scholi/pySPM) to extract metadata from IONTOF `.itm` and `.ita` files.
The main input file is .itm but requires a `.grd` raw data file in the same folder, which need to be exported with the Surfacelab tool ITRawExport. 
Future versions will aim to incorporate also .imzML inputs and custom tabular inputs.

#### SIMSpy function overview 

SIMSpy has three routines.
1. Untargeted multivariate analysis `SIMSpy_MVA.py`  <br>
2. Molecular formula prediction `SIMSpy_MFP.py` <br>
3. Targeted analsysis `SIMSpy_Targeted.py`  <br>


## 1. Untargeted multivariate analysis

SIMSpy can reveal hidden patterns in data using statistical analysis. Each step is fully automated, enabling
ROI detection and untargeted depth profile extraction without manual software use.
We list the key arguments for each of the main steps:


### Binning

Binning reduces the size of a dataset by summing neibouring pixels, scans, or mass bins.
This also reduces noise. Additionally, files can be truncated before binning, to only contain a certain range of pixels, scans or masses.
For high spatial resolution fast imaging datasets, larger values for bin_pixels are recommended.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|min_mass, max_mass| 0,1000 | mass range (Da)|       
|min_x, max_x, min_y, max_y, min_scans, max_scans |0 |  truncate in 3D space|
|bin_pixels, bin_scans, bin_tof| 0,0,0 | merge neigboring pixels, scans, tof values|

### Peak picking
Peak picking is used for calibration, depth profile extraction, and for detecting mass resolution and ROIs.
Peak detection only works when there is sufficient mass resolution.
Since TOF-SIMS peaks can overlap due to poor mass separation, 
peaks can be deconvoluted with gaussian mixture modellling.

### Calibration 

Calibration is key for accurate molecular formula prediction.
Using a list of internal calibrants, 
first optimal values for global sf and k0 are fitted, which as used to convert tof values to mz,
then calibration minimizes the mass error locally.
There is a list of substrate specific calibrants, and a list of sample specific calibrants.
If calibration is unsuccesful, a higher ppm_cal value should be tried.
Sor gold coated silicon wafers, a substrate value of Au would be used. ITO substrates, InOSiSn can be used.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|ppm_cal| 100 | Mass error tolerance for detecting internal calibrants|     
|Calibrants | Calibrants.csv                     |List of typical internal calibrants |
|Substrate_Calibrants | Substrate_Calibrants.csv  |List of internal calibrants coming from the substrate|
|Substrate|Au |  List of elements present in substrate (filters substrate internal calibrants to only contain these elements) |

### Mass resolution detection

Fitting the mass resolution can be used to predict the FWHM of peaks as a function of mass.
This can then be used for peak deconvolution. Optionanly too narrow or too wide peaks can be removed.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|min_width_ratio,max_width_ratio| 0  |remove based on ratio e of measured peak width/ expected peak width|

### ROI detection

Regions of interest (ROI) are detected using Kmeans clustering, in 2D or 3D space to perform image segmentaiton.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|ROI_bin_pixels,ROI_bin_scans| 2,2| additional binning for ROI |
|ROI_clusters| 3 | Number of regions |     
|ROI_dimensions | 3                     | 2D or 3D |
|ROI_scaling|"Jaccard" |  Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"|


### Depth profile extraction

Depth profiles foreach peak are exported per ROI. They can be smoothed with moving average, or normalized to total.
|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|normalize| True | noramalize to total |     
|smoothing | 3                     | weighted average smoothing window|

### Isotope detection
To improve molecular formula predictio, SIMSpy_MVA includes isotope detection.
For each mass peaks, potential isotope peaks are detected with a certain mass error.
Candidate isotope peaks are filtered on their cosine similarity.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|isotope_range| [-4,6] | minimum isotope, maximimum isotope |     
|min_cosine | 0.9                     | minimum cosine similarity to mono isotope|


### MVA analsysis
SIMSpy_MVA allows for multivarariate analsysis (PCA or NMF) in up to 3 dimensions.
MVA can either be applied to detected peaks, or directly to mass bins. This 2nd option results in large array size and typically requires larger bin_TOF values. 
For 1D, MVA can be applied to the depth profiles of each ROI.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|data_reduction| "binning" | "binning" or "peak_picking" |
|MVA_bin_pixels,MVA_bin_scans,MVA_bin_tof| 2,2, 5| additional binning for MVA |
|MVA_components| 3 | Number of regions |     
|MVA_methods| ["NMF","PCA"]                    | 2D or 3D |
|MVA_dimensions|"Standard" |  Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"|



## 2. Molecular formula prediction

Building a database
Molecular formula prediction
Isotope filtering
Calibration

## 3. Molecular formula prediction

Building a database
Molecular formula prediction
Isotope filtering
Calibration


## Example use 
Molecular formula prediction from command line:
``` 
python "cart2form.py" -input_file "test_mass_CASMI2022.txt" -composition_file "H[200]C[75]N[50]O[50]P[10]S[10]_b100000max1000rdbe-5_80_7gr_comp.npy"
```
Alternatively cart2form.py can be imported.
``` 
import cart2form
cart2form.predict_formula(input_file=124.56 ,   #float mass or iterable (array/list/DataFrame)
                          composition_file="H[200]C[75]N[50]O[50]P[10]S[10]_b100000max1000rdbe-5_80_7gr_comp.npy")
```


#### Licensing

The pipeline is licensed with standard MIT-license. <br>
If you would like to use this pipeline in your research, please cite the following papers: 

#### Contact:
-Hugo Kleikamp (Developer): hugo.kleikamp@uantwerpen.be<br> 



#### Related repositories:
https://github.com/scholi/pySPM<br>
https://github.com/hbckleikamp/CartMFP
https://github.com/hbckleikamp/HorIson

