<h1 align="center">
<img src="https://raw.githubusercontent.com/fredericjs/surfalize/refs/heads/main/logo.svg" width="600">
</h1><br>

[![PyPI version](https://badge.fury.io/py/surfalize.svg)](https://badge.fury.io/py/surfalize)
[![Documentation Status](https://readthedocs.org/projects/surfalize/badge/?version=latest)](https://surfalize.readthedocs.io/en/latest/?badge=latest)
[![Publication](https://img.shields.io/badge/Publication%20-%2010.3390%2Fnano14131076%20-%20%20%2385c1e9%20
)](https://doi.org/10.3390/nano14131076)



surfalize is a python package for analyzing microscope topography measurement data in terms of surface
roughness and other topographic parameters. It is intended primarily for microtextured surfaces and is supposed to 
replace software packages such as MountainsMap, MultiFileAnalyzer and Gwyddion for the most common tasks.

## Disclaimer
The authors make no guarantees for the correctness of any results obtained using this package. The package is an early work in progress
and may introduce changes to both implementation details and public API at any point in time. Any results should be validated against established 
software to verify their correctness, especially when they are intended to be used for scientific publications. 

Some parts of the package are more mature and some are in early development stage. Currently, Gaussian filtering and Profile parameters might 
suffer from some implementation errors and might not be entirely compliant with ISO standards. Care should be taken when relying on these
specific functionalities.

## Cite this library

If you publish data generated with this library, please consider citing this publication:

[F. Schell, C. Zwahr, A. F. Lasagni. Surfalize: A Python Library for Surface Topography and Roughness Analysis Designed 
for Periodic Surface Structures. _Nanomaterials_. **2024**, 13, 1076.](https://doi.org/10.3390/nano14131076)

## How to install

To install the latest release of surfalize, run the following command: 
```commandline
pip install surfalize
```
If you want to install from source, clone this git repository and run the following command in the root folder
of the cloned repository.
```commandline
pip install .
```
While earlier versions used Cython, the current version is pure Python and can be installed without a C-compiler.

## Documentation

The documentation is hosted on [readthedocs](https://surfalize.readthedocs.io/en/latest/).

## Currently supported file formats

| Manufacturer           | Format                 | Reading         | Writing |
|------------------------|------------------------|-----------------|---------|
| Keyence                | *.vk4*, *.vk6*, *.vk7* | Yes             | No      | 
| Keyence                | *.cag*                 | Only extraction | No      | 
| Leica                  | *.plu*                 | Yes             | No      | 
| Sensofar               | *.plu*, *.plux*        | Yes             | No      | 
| Digital Surf           | *.sur*                 | Yes             | Yes     | 
| KLA Zeta               | *.zmg*                 | Yes             | No      | 
| Wyko                   | *.opd*                 | Yes             | No      | 
| Nanofocus              | *.nms*                 | Yes             | No      | 
| Alicona                | *.al3d*                | Yes             | Yes     | 
| Digital Surf           | *.sdf*                 | Yes             | Yes     | 
| Gwyddion               | *.gwy*                 | Yes             | No      | 
| Digital Metrology      | *.os3d*                | Yes             | No      |
| IAU FITS Working Group | *.fits*                | Yes             | No      |
| General                | *.xyz*                 | Yes             | No      |

## Supported roughness parameters

This package aims to implement all parameters defined in ISO 25178. Currently, the following parameters are supported:

| Category            | Parameter       | Full name                         | Validated against                  |
|---------------------|-----------------|-----------------------------------|------------------------------------|
| Height              | Sa              | Arithmetic mean height            | Gwyddion, MountainsMap             |
|                     | Sq              | Root mean square height           | Gwyddion, MountainsMap             |
|                     | Sp              | Maximum peak height               | Gwyddion, MountainsMap             |
|                     | Sv              | Maximum valley depth              | Gwyddion, MountainsMap             |
|                     | Sz              | Maximum height                    | Gwyddion, MountainsMap             |
|                     | Ssk             | Skewness                          | Gwyddion, MountainsMap             |
|                     | Sku             | Kurtosis                          | Gwyddion, MountainsMap             |  
| Hybrid              | Sdr<sup>1</sup> | Developed interfacial area ratio  | Gwyddion<sup>2</sup>, MountainsMap |
|                     | Sdq             | Root mean square gradient         | MountainsMap                       |
| Spatial             | Sal             | Autocorrelation length            | -                                  |
|                     | Str             | Texture aspect ratio              | -                                  |
| Functional          | Sk              | Core roughness depth              | MountainsMap                       |
|                     | Spk             | Reduced peak height               | MountainsMap                       |
|                     | Svk             | Reduced dale height               | MountainsMap                       |
|                     | Smr1            | Material ratio 1                  | MountainsMap                       |
|                     | Smr2            | Material ratio 2                  | MountainsMap                       |
|                     | Sxp             | Peak extreme height               | MountainsMap                       |
| Functional (volume) | Vmp             | Peak material volume              | MountainsMap                       |
|                     | Vmc             | Core material volume              | MountainsMap                       |
|                     | Vvv             | Dale void volume                  | MountainsMap                       |
|                     | Vvc             | Core void volume                  | MountainsMap                       |

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