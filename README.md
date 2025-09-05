# SIMSpy

SIMSpy is a collection of python routines that enable untargeted multivariate analysis of TOF-SIMS data, molecular formula prediction, and targeted analysis of fragments.
it relies on the module pySPM (github.com/scholi/pySPM) to extract metadata from IONTOF `.itm` and `.ita` files.
The main input file is .itm but requires a `.grd` raw data file in the same folder, which need to be exported with the Surfacelab tool ExportITMtoGRD. 
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

### Inputs

As inputs  a list of .itm files is used. The script will automatically look for the corresponding .grd files.
If they are not present, it will try to export them using the path to the grd_exe.
|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|itmfiles|  | list of files or folder with .itm files|     
|grd_exe | ITRawExport.exe                  |full path to executable |

### Binning

Binning reduces the size of a dataset by summing neibouring pixels, scans, or mass bins.
This also reduces noise. Additionally, files can be truncated before binning, to only contain a certain range of pixels, scans or masses.
For high spatial resolution fast imaging datasets, larger values for bin_pixels are recommended.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|min_mass, max_mass| 0,1000 | mass range (Da)|       
|min_x, max_x, min_y, max_y, min_scans, max_scans |0 |  truncate in 3D space|
|bin_pixels, bin_scans, bin_tof| 0 | merge neigboring pixels, scans, tof values|

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
|MVA_methods| ["NMF","PCA"]                    | NMF or PCA |
|MVA_dimensions|[1,2,3]|  1,2 or 3 dimensions |
|MVA_scaling|"Standard" |  Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"|


## 2. Molecular formula prediction

Molecular formula prediction is based on the tool CartMFP (github.com/hbckleikamp/CartMFP), while isotope simlation is done using HorIson (github.com/hbckleikamp/HorIson).
As inputs for MFP exported depth profiles or MVA loadings from SIMSpy_MVA can be used.

### Input

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|input_files|  | list of depth profiles or list of masses |
|MFP_dbf| |  path to database file as constructed by space2cart.py|


### Building a database
The database architecture of CartMFP is used, this includes using `space2cart.py` for database construction.
This constructs a local database with all possible combinations of elements within a certain compositional space.
See the documentation of CartMFP for a more in depth explanation of the syntax.

### Molecular formula prediction
Molecular formula predition is done using the constructed database. Additionally, a list of background ions can be supplied,
which are also used as internal calibrants.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|ppm| 500 | max mass error tolerance of predicted formulas |
|top_candidates| 100|  number of top canidadates in output|


### Calibration 
After the first round of molecular formula prediction, formulas that match internal calibrants are used to recalibrate the masses within the dataset.
A secondary MFP is done after recalibration with a more narrow ppm window (pos_calib_ppm)

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|post_calib_ppm| 200 | Mass error tolerance for secondary MFP after recalibration|     
|Calibrants | Calibrants.csv                     |List of typical internal calibrants |
|Substrate_Calibrants | Substrate_Calibrants.csv  |List of internal calibrants coming from the substrate|
|Substrate|Au |  List of elements present in substrate (filters substrate internal calibrants to only contain these elements) |

### Isotope filtering ###
Isotope filtering provides a way to improve the quality of annotations.
Since TOF-SIMS lacks the mass resolution to distinguish +H mass from +n, isotope filtering is only useful to highlight missing isotopes.
For earch predicted molecualr formula isotopes are predicted within a certain range using HorIson.
Isotope intensity is normalized towards the monoisotopic peak. Isotopes below a certain intensity treshold are removed.
Remaining expected isotopes are compared to the measured isotopic envelope, using the cosine similarity.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
isotope_range| [-2,6] | minimum, maximimum range of simulated isotopes |     
|min_intensity | 10                    |minimum intensity treshold of simulated isotopes |

### ppm filtering
As secondary qualtiy metric (next to cosine similarity) the local average ppm error is calculated.
Correct formula predictions are expected to have similarl ppm mass errors.
This is expressed with the dppm value (delta ppm).

## 3. Targeted analysis

While SIMSpy_MVA is suitable for untargeted analysis, SIMSpy_Targeted can handle experiments that want to analyze the distribution of a set of known fragments.
Similar to SIMSpy_MVA, the pipeline first performs a set of operations, such as Truncating, Calibration, Mass resolution detection and Peak deconvolution, which use the same syntax.
### Inputs

As inputs  a list of .itm files is used. The script will automatically look for the corresponding .grd files.
If they are not present, it will try to export them using the path to the grd_exe.
|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|itmfiles|  | list of files or folder with .itm files|     
|grd_exe | ITRawExport.exe                  |full path to executable |
|Target | Targets.csv |list of known fragments |

### Extract targets
The first unique part is the extraction of target fragments, which are supplied as a list.
The abundance of the fragments can be plotted individually in 1,2 or 3 dimensions, or as a set within a group.
While group labels can be supplied in the input, they can also be discovered automatically

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|ppm| 100 | maximum mass error for detecting known fragments|     
|Target_bin_pixels,Target_bin_scans| 2,2                |merging of neighboring pixels or scans|
|Target_dimension| 2 |which dimension to plot [1,2, or 3] |
|Sum_by_Groups | True | instead of plotting each fragment individually, sum them by their group label |

### Group detection
While groups can be supplied manually, fragments can also be grouped on their abundance across datapoints.
To do this, first a PCA is performed in 2 dimensions (x,y) followed by hierachrical clustering of the PCA components based on cosine distance.
After extracting PCA, additional untargeted analysis can be included by adding unidentified peaks to the clustering step.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|Determine_groups| True | overwrite added group labels with PCA and clustering |     
|Scaling| Standard  | Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"||
|n_components| 2 |number of PCA components|
|cluster_distance| 0.6 | value between 0-1 or "auto"|
|Expand_groups| 0 |number unidentified peaks to include in clustering|

### Pairwise comparison
After different groups have been detected, the difference in counts between each group can be calculated.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|pairwise_diffs| True | compute distance between groups |     
|maximize_difference| True  | scale groups to maximize small differences |
|correlate| True | calculate cosine similarity between groups|

## Example use 
Each pipeline can be executed within an IDE, or from the command line, below are examples on CLI usage.

SIMSpy_MVA for ROI detection (4) regions in 3D
``` 
python "SIMSpy_MVA.py" -itmfiles "test.itm" -ROI_clusters 4 -ROI_dimensions 3 
```
SIMSpy_MFP for molecular formula prediction with a custom database
``` 
python "SIMSpy_MFP.py" -input_files "test.itm" -MFP_db "my_db.comp" -ppm 100 -top_candidates 20
```
SIMSpy_Targeted for plotting groups of known fragments in 3D
``` 
python "SIMSpy_Targeted.py" -itmfiles "test.itm"  -Target "my_fragments.csv" -Target_dimension 3
```


#### Licensing

The pipeline is licensed with standard MIT-license. <br>
If you would like to use this pipeline in your research, please cite the following papers: (future placeholder)

#### Contact:
-Hugo Kleikamp (Developer): hugo.kleikamp@uantwerpen.be<br> 



#### Related repositories:
https://github.com/scholi/pySPM<br>
https://github.com/hbckleikamp/CartMFP<br>
https://github.com/hbckleikamp/HorIson

