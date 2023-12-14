<h1 align="center">
<img src="logo.svg" width="600">
</h1><br>

surfalize is a python package for analyzing microscope topography measurement data in terms of surface
rouggness and other topographic parameters. It is intended primarily for microtextured surfaces and is supposed to 
replace software packages such as MountainsMap, MultiFileAnalyzer and Gwyddion for the most common tasks.

## How to install

To install the latest release of surfalize, run the following command: 
```commandline
pip install surfalize
```
If you want to build from source, clone this git repository and run the following command in the root folder
of the cloned repository.
```commandline
pip install .
```
However, you will need to have both `Cython` and a C-Compiler installed (MSVC on Windows, 
gcc on Linux, MinGW is not supported currently). If you install in editable mode using
```
pip install -e .
```
be aware that a change of the pyx files does not reinvoke the Cython build process. 


## Currently supported file formats

| Manufacturer | Format                 |
|--------------|------------------------|
| Keyence      | *.vk4*, *.vk6*, *.vk7* |
| Leica        | *.plu*                 |
| Sensofar     | *.plu*, *.plux*        |
| Digital Surf | *.sur*                 |

## Supported roughness parameters

This package aims to implement all parameters defined in ISO 25178. Currently, the following parameters are supported:

| Category            | Parameter       | Full name                        | Validated against                  |
|---------------------|-----------------|----------------------------------|------------------------------------|
| Height              | Sa              | Arithmetic mean height           | Gwyddion, MountainsMap             |
|                     | Sq              | Root mean square height          | Gwyddion, MountainsMap             |
|                     | Sp              | Maximum peak height              | Gwyddion, MountainsMap             |
|                     | Sv              | Maximum valley depth             | Gwyddion, MountainsMap             |
|                     | Sz              | Maximum height                   | Gwyddion, MountainsMap             |
|                     | Ssk             | Skewness                         | Gwyddion, MountainsMap             |
|                     | Sku             | Kurtosis                         | Gwyddion, MountainsMap             |  
| Hybrid              | Sdr<sup>1</sup> | Developed interfacial area ratio | Gwyddion<sup>2</sup>, MountainsMap |
|                     | Sdq             | Root mean square gradient        | MountainsMap                       |
 | Spatial             | Sal             | Autocorrelation length           | -                                  |
|                     | Str             | Texture aspect ratio             | -                                  |
| Functional          | Sk              | Core roughness depth             | MountainsMap                       |
|                     | Spk             | Reduced peak height              | MountainsMap                       |
|                     | Svk             | Reduced dale height              | MountainsMap                       |
|                     | Smr2            | Material ratio 1                 | MountainsMap                       |
|                     | Smr1            | Material ratio 2                 | MountainsMap                       |
|                     | Sxp             | Peak extreme height              | MountainsMap                       |
| Functional (volume) | Vmp             | Peak material volume             | MountainsMap                       |
|                     | Vmc             | Core material volume             | MountainsMap                       |
|                     | Vvv             | Dale void volume                 | MountainsMap                       |
|                     | Vvc             | Core void volume                 | MountainsMap                       |

<sup>1</sup> Per default, Sdr calculation uses the algorithm proposed by ISO 25178 and also used by MountainsMap
By keyword argument, the Gwyddion algorithm can be used instead.\
<sup>2</sup> Gwyddion does not support Sdr calculation directly, but calculates surface area and projected
area. 

## Supported parameters of 1d-periodic surfaces

Additionally, this package supports the calculation of non-standard parameters for periodic textured surfaces with one-
dimensional periodic structures. The following parameters can be calculated:

| Parameter    | Description                                                  |
|--------------|--------------------------------------------------------------|
| Period       | Dominant spatial period of the 1d-surface texture            | 
| Depth        | Peak-to-valley depth of the 1d-texture profiles              | 
| Aspect ratio | Ratio of peak-to-valley depth to spatial period              |
| Homogeneity  | Homogeneity factor (0 < H < 1) calculated from Gini analysis |
| Orientation  | Clockwise angle of the dominant texture to the vertical axis |

## Supported operations

| Operation       | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| Leveling        | Subtraction of least squares fit to a plane                        | 
| Zeroing         | Sets the lowest datapoint of the surface to zero                   | 
| Centering       | Sets the average value of the surface elevation to zero            |
| Zooming         | Magnifies the surface by a specified factor                        |
| Cropping        | Crops the surface in a specified rectangle                         |
| Rotation        | Rotates the surface by a specified angle in degrees                |
| Alignment       | Aligns the surface with the dominant texture direction by rotation |
| Outlier removal | Removes outliers outside n standard deviation from the mean        |
| Thresholding    | Thresholding based on areal material ratio                         |
| Filtering       | Applies a Gaussian highpass, lowpass or bandpass filter            |

## Basic Usage

```
from surfalize import Surface

filepath = 'example.vk4'
surface = Surface.load(filepath)
surface.show()
```

### Extracting roughness and topographic parameters

```
# Individual calculation of parameters
sa = surface.Sa()
sq = surface.Sq()

# Calculation in batch
parameters = surace.roughness_parameters(['Sa', 'Sq', 'Sz'])
>>> parameters
{'Sa': 1.25, 'Sq': 1.47, 'Sz': 2.01} 

# Calculation in batch of all available parameters
all_available_parameters = surace.roughness_parameters()
```

### Performing surface operations

```
# Levelling the surface using least-sqaures plane compensation

# Returns new Surface object 
surface = surface.level()

# Returns self (to allow for method chaining)
surface.level(inplace=True)

# Filters the surface using the Fourier transform
surface_low = surface.filter(10, mode='highpass', inplace=False)

# Filter form and noise
surface_filtered = surface.filter(0.8, mode='bandpass', cutoff2=10)

# # Separate waviness and roughness and return both
surface_roughness, surface_waviness = surface.filter(10, mode='both')

# If the surface contains any non-measured points, the points must be interpolated before any other operation can be applied
surface = surface.fill_nonmeasured(mode='nearest')

# The surface can be rotated by a specified angle in degrees
# The resulting surface will automatically be cropped to not contain any areas without data
surface.rotate(10)
```

These methods can be chained:

```
surface = Surface.load(filepath).level().filter(filter_type='lowpass', cutoff=0.8)
surface.show()
```

### Batch processing

```
from pathlib import Path
from surfalize import Batch

filepaths = Path('folder').glob('*.vk4')

# Create a Batch object that holds the filepaths to the surface files
batch = Batch(filepaths)
```

All operations of the surface can be applied to the Batch analogously to a Surface object.
However, they are not applied immediately but registered for later execution.
```
batch.level()
batch.filter('highpass', 20)
```

Each operation on the batch returns the Batch object, allowing for method chaining.
```
batch = Batch(filepaths).level().filter('highpass', 20).align().center()
```
The calculation of roughness parameters can be done indiviually and chained.
```
batch.Sa().Sq().Sq().Sdr()
```
Arguments to the roughness parameter calculations, such as _p_ and _q_ can be provided in the individual call.
```
batch.Vmc(p=10, q=80)
```
Parameters can also be calculated in bulk using `Batch.roughness_parameters()`:
```
# Computes Sa, Sq, Sz
batch.roughness_parameters(['Sa', 'Sq', 'Sz'])
# Computes all available parameters
batch.roughness_parameters()
```
If arguments need to be supplied, the parameter must be constructed as a `Parameter` object:
```
from surfalize.batch import Parameter
Vmc = Parameter('Vmc', kwargs=dict(p=10, q=80))
batch.roughness_parameters(['Sa', 'Sq', 'Sz', Vmc])
```

Finally, the batch processing is executed by calling `Batch.execute`, returning a `pd.DataFrame`. Optionally, 
`multiprocessing=True` can be specified to split the load among all available CPU cores. Moreover, the results 
can be saved to an Excel Spread sheet by specifiying a path for `saveto=r'path\to\excel_file.xlsx`.
```
df = batch.execute(multiprocessing=True)
```

Optionally, a Batch object can be initialized with a filepath pointing to an Excel File which contains additional
parameters, such as laser parameters. The file must contain a column `file`, which specifies the filename including file
extension in the form `name.ext`, e.g. `topography_50X.vk4`. All other columns will be merged into the resulting
Dataframe that is returned by `Batch.execute`. 

```
batch = Batch(filespaths, additional_data=r'C:\users\exampleuser\documents\laserparameters.xlsx')
batch.level().filter('highpass', 20).align().roughness_parameters()
df = batch.execute()
```

Full example: Let's supppose we have four topography files called `topo1.vk4`, `topo2.vk4`, `topo3.vk4`, `topo4.vk4` in 
the folder `C:\users\exampleuser\documents\topo_files`. Moreover, we have additional information on these files in an 
Excel files located in `C:\users\exampleuser\documents\topo_files\laserparameters.xlsx`. The Excel looks like this:


| file      | power | pulse_overlap | hatch_distance |
|-----------|-------|---------------|----------------|
| topo1.vk4 | 100   | 20            | 12.5           |
| topo2.vk4 | 50    | 20            | 12.5           |
| topo3.vk4 | 100   | 50            | 12.5           |
| topo4.vk4 | 50    | 50            | 12.5           |

```
from pathlib import Path
from surfalize import Batch

filepaths = Path(r'C:\users\exampleuser\documents\topo_files').glob('*.vk4')
batch = Batch(filespaths, additional_data=r'C:\users\exampleuser\documents\topo_files\laserparameters.xlsx')
batch.level().filter(20, mode='highpass').align().roughness_parameters(['Sa', 'Sq', 'Sz'])
batch.execute(multiprocessing=True, saveto=r'C:\users\exampleuser\documents\roughness_results.xlsx')
```

The result will be a DataFrame that looks like this:

| file      | power | pulse_overlap | hatch_distance | Sa   | Sq   | Sz   |
|-----------|-------|---------------|----------------|------|------|------|
| topo1.vk4 | 100   | 20            | 12.5           | 0.85 | 1.25 | 3.10 | 
| topo2.vk4 | 50    | 20            | 12.5           | 0.42 | 0.51 | 1.87 | 
| topo3.vk4 | 100   | 50            | 12.5           | 1.34 | 1.67 | 3.84 | 
| topo4.vk4 | 50    | 50            | 12.5           | 0.55 | 0.67 | 1.99 | 