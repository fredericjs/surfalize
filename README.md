<h1 align="center">
<img src="logo.svg" width="600">
</h1><br>

surfalize is a python package for analyzing microscope topography measurement data in terms of surface
rouggness and other topographic parameters. It is intended primarily for microtextured surfaces and is supposed to 
replace software packages such as MountainsMap, MultiFileAnalyzer and Gwyddion for the most common tasks.

## Currently supported file formats

| Manufacturer | Format                 |
|--------------|------------------------|
| Keyence      | *.vk4*, *.vk6*, *.vk7* |
| Leica        | *.plu*                 |
| Sensofar     | *.plu*, *.plux*        |

## Supported roughness parameters

| Category   | Parameter       | Full name                        | Validated against                  |
|------------|-----------------|----------------------------------|------------------------------------|
| Height     | Sa              | Arithmetic mean height           | Gwyddion, MountainsMap             |
|            | Sq              | Root mean square height          | Gwyddion, MountainsMap             |
|            | Sp              | Maximum peak height              | Gwyddion, MountainsMap             |
|            | Sv              | Maximum valley depth             | Gwyddion, MountainsMap             |
|            | Sz              | Maximum height                   | Gwyddion, MountainsMap             |
|            | Ssk             | Skewness                         | Gwyddion, MountainsMap             |
|            | Sku             | Kurtosis                         | Gwyddion, MountainsMap             |      |
| Hybrid     | Sdr<sup>1</sup> | Developed interfacial area ratio | Gwyddion<sup>2</sup>, MountainsMap |
|            | Sdq             | Root mean square gradient        | MountainsMap                       |
| Functional | Sk              | Core roughness depth             | MountainsMap                       |
|            | Spk             | Reduced peak height              | MountainsMap                       |
|            | Svk             | Reduced dale height              | MountainsMap                       |
|            | Smr2            | Material ratio 1                 | MountainsMap                       |
|            | Smr1            | Material ratio 2                 | MountainsMap                       |

<sup>1</sup> Deviations in Sdr are possible due to chosen algorithm\
<sup>2</sup> Gwyddion does not support Sdr calculation directly, but calculates surface area and projected
area

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
surface = Surface.load(filepath).level().filter(0.8, mode='lowpass')
surface.show()
```

### Batch processing

```
from pathlib import Path
from surfalize import Batch

filepaths = Path('folder').glob('*.vk4')

batch = Batch(filepaths)
# Returns pandas DataFrame with the calculated parameters
df = batch.roughness_parameters(['Sa', 'Sq', 'Sz'])

# Surface operations can also be applied to the batch in the same way as for a single surface
# Note that the operations are inplace by default
df = batch.level().filter(0.8, mode='highpass').roughness_parameters()
```
