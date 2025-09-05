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
|bin_pixels, bin_scans, bin_tof| 2,2,5 | merge neigboring pixels, scans, tof values|

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

#### Mass resolution detection

Fitting the mass resolution can be used to predict the FWHM of peaks as a function of mass.
This can then be used for peak deconvolution. Optionanly too narrow or too wide peaks can be removed.

|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|min_width_ratio,max_width_ratio| 0  |remove based on ratio e of measured peak width/ expected peak width|

#### ROI detection

Regions of interest (ROI) are detected using Kmeans clustering, in 2D or 3D space to perform image segmentaiton.


|Parameter           | Default values     |       Description|
|-----------------|:-----------:|---------------|
|ROI_clusters| 3 | Number of regions |     
|ROI_dimensions | 3                     | 2D or 3D |
|ROI_scaling|"Poisson" |  Options: False, "Poisson", "MinMax", "Standard", "Robust", "Jaccard"|


#### Depth profile extraction

Depth profiles for each individual ROI are exported. 


Smoothing
Normalize

#### Isotope detection

Isotope range
Cosine correlation

#### MVA analsysis
1D MVA can be applied to the depth profiles of each ROI.

MVA type
components
dimensions



## 2. Molecular formula prediction

Building a database
Molecular formula prediction
Isotope filtering
Calibration

## 2. Molecular formula prediction


A database is constructed by enumerating all combinations of elements within a certain range.
The compositional space is described with a specific syntax: Element[min,max].
This can be any element in the periodic table for which the monoisotopic mass is described in the NIST database.
"H[200]C[75]N[50]O[50]P[10]S[10]" is used default elemental space.
This would eqaute to 0-200 Hydrogen, 0-75 Carbon, 0-50 Nitrogen, etc. 

Apart from max element constraints, the elemental composition space is further limited by the maximum mass `max_mass` and ring double bond equivalents (RDBE) `min_rdbe`,`max_rdbe`.

Base chemical constraints:
|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-composition| "H[200]C[75]N[50]O[50]P[10]S[10]" | composition string describing minimum and maximum element counts|
|-max_mass| 1000 | maximum mass (Da)|
|-min_rdbe | -5 | minimum RDBE |
|-max_rdbe| 80 | maximum RDBE |


Additional chemical constraints are provided by implementing some of Fiehn's 7 Golden rules, which filters unrealistic or impossible compositions.
This can drastically reduce the size of your composition space. These include:  Rule #2 – LEWIS and SENIOR check; Rule #4 – Hydrogen/Carbon element ratio check; Rule #5 heteroatom ratio check and Rule #6 – element probability check.

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-filt_7gr| True | Toggle global to apply or remove 7 golden rules filtering|
|-filt_LewisSenior| True | Golden Rule  #2:   Filter compositions with non integer dbe (based on max valence) |
|-filt_ratios | "HC[0.1,6]FC[0,6]ClC[0,2]BrC[0,2]NC[0,4]OC[0,3]PC[0,2]SC[0,3]SiC[0,1]" | #Golden Rules #4,5: Filter on chemical ratios with extended range 99.9% coverage |
|-filt_NOPS| True    | #6 – element probability check. |

Additional arguments can be supplied to affect the performance and output paths: 

|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-maxmem | 10e9 |  Amount of memory used in bytes |
|-mass_blowup | 100000 |blowup factor to convert float masses to integers|
|-write_mass  | True | construct a mass lookup table (faster MFP but larger database)|
|-Cartesian_output_folder | "Cart_Output" | Path to output folder |
|-Cartesian_output_file   |<depends on parameters> | Output database name |

## Example use space2cart (In command line)

Contstruct default database:
``` 
python "space2cart.py" 
```

Contstruct database with halogens:
``` 
python "space2cart.py" -composition "H[200]C[75]N[50]O[50]P[10]S[10]F[5]Cl[5]I[3]Br[3]"
```
space2cart can also be executed by running the script in an IDE, such as Spyder.

# 2. Molecular formula prediction

After the composition database has been constructed with `space2cart.py` , molecular formula prediction can be done using `cart2form.py`.
To run cart2form, an input mass list has to be supplied, which can be linked to a file in txt or any tabular format.
The file either has to have a single column, or a column titled  "Mz" or "Mass".
Alternatively cart2form can be imported as a module within a script, and executed on a float mass or iterable set of masses with the function `predict_formula(input_file,composition_file)`.

#### required inputs 
|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-input_file       | "test_mass_CASMI2022.txt"| path to list of masses|
|-composition_file | "H[200]C[75]N[50]O[50]P[10]S[10]_b100000max1000rdbe-5_80_7gr_comp.npy"| path to the database composition file|
                           

Optional arguments can supplied to tune which massses will be returned, this includes
the polariy, which adducts to consider, which charge states to consider.
Other key arguments include the ppm tolerance of the mass error of returned composition, and the maximum number of compositions returned per mass.  Adducts use the following syntax: "sign(+ addition of /- loss of) elemental composition charge(+/-)"
|Parameter           | Default value     |       Description|
|-----------------|:-----------:|---------------|
|-mode | "pos"                     | ionization mode. Options: " ", "positive", "negative"|
|-adducts|["+H+","+Na+","+K",   "+-","+Cl-","-H+"]       | default positive adducts "--","+H+","+Na+","+K", default negative adducts "+-","-H+","+Cl-" |
|-charges|[1]                         |Charge states to consider|
|-ppm| 5| maximum mass error (ppm) of predicted compositions |
|-top_candidates | 20 |maxmimum number of compositions returned per mass|


## Example use cart2form
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
-Martin Pabst (Co-Developer): m.pabst@tudelft.nl<br>


#### Related repositories:
https://github.com/bbuchfink/diamond<br>
https://github.com/hbckleikamp/proteomic-database-prepper<br>
https://github.com/hbckleikamp/NCBI2Lineage<br>
https://github.com/hbckleikamp/GTDB2DIAMOND<br>
https://github.com/hbckleikamp/De-Novo-ErrorSIM


