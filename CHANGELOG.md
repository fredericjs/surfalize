## v0.15.0
- Added 3d plotting cababilities based on pyvista
- Added pyvista as optional dependency for 3d plotting
- Fixed bug in AutocorrelationFunction plotting
- Added Zygo Metropro .dat format reader
## v0.14.2
- Fixed issues with Python < 3.10 and on linux
## v0.14.1
- Fixed roughness parameter calculation based on ACF (Sal, Str)
- Added interpolation to the calculation ACF roughness parameters, now matching values obtained by MountainsMap more
  closely
- Fixed broken link to logo for PyPi
## v0.14.0
- Refactored the file reading and writing system
- Readers and writers now register in a FileHandler class with their suffix and file magic
- Readers and writers now work directly with file-like objects
- If the designated reader for a file determined from the file path suffix raises an Exception, if no suffix is provided
  or if no reader exists for that suffix, surfalize now tries to infer the correct reader from the file magic. This 
  helps in cases where the file format is unknown or the file is labeled with the wrong file extension.
- Added tests for reading from and writing to file-like objects
- File loader now issues warning if step_x and step_y are not equal
- Added support for file-like objects to batch processing
- Updated docs for working with file-like objects
## v0.13.2
- Ensured compatibility of trapezoid function with numpy versions (trapz for np < 2.X, trapezoid for > 2.X) 
- Added hook to batch execution that is called after each processed file. This hook can be used for instance to display
  a custom progress bar.
## v0.13.1
- Updated minimum Python version requirement to 3.9
- Fixed bug with OPD file reader that occurs when no date is present in the metadata
## v0.13.0
- Added support for extracting cag files by importing extract_cag from surfalize.file.cag (currently undocumented)
- Added support for adding custom parameters to batch calculation through Batch.custom_parameter 
## v0.12.1
- Fixed bug where Surface.level would return only nan if surface has missing points
## v0.12.0
- Fixed string match in filename extraction to also include non alphabetic characters
- Batch.execute now returns BatchResult class instead of DataFrame, which wraps the DataFrame and exposes the same
  methods, but allows for other methods such as filename extraction on the result and potentially other methods in the
  future
- Fixed bug in orientation_fft
- Added vmin, vmax and show_cbar keyword to Surface.plot_2d
- Added SFLZ file format
- Refactored file layout reading and writing
- Replaced DataFrame with BatchResult as the return value for Batch.execute. BatchResult wraps the DataFrame but also 
  exposes additional methods, such as filename parameter extraction.
- Batch.execute now raises error if parameter is computed twice and overwritten
- Parameters in Batch can now be given a custom name using the custom_name kwarg to change the name of the column in
  the DataFrame. This allows for the computation of multiple versions of the same parameter
- Removed numpy dependency restriction to versions below 2.0
## v0.11.1
- Removed hierarch keyword from fits readers
- Fixed bug with reshaping of fits arrays
## v0.11.0
- Added polynomial detrending
- Added FITS file format
## v0.10.2
- Fixed bug with file saving
- Fixed bug with saving to SUR file format
- Fixed bug with inplace removal of outliers, where the surface's non_measurepoints attribute was not updated, causing 
  filling of missing points to return without being executed
- Added tests for fileformat writing
## V0.10.1
- Entirely removed Cython from the codebase and replaced with efficient numpy code. This allows for the distribution
  of this library without compilation for different architectures and operating systems.
## v0.10.0
- Added ascii sdf file format
- Added ignore_errors kwarg to Batch.execute to ignore errors during parameter calculation and fill the respective 
  values with nan instead
- Added writing of ascii and binary sdf file format
- Removed skipping of values in binary layouts
- Refactored ACF class, which now supports plotting
- Surface objects now have public methods to get the ACF and Abbott-Firestone curve
- Equal comparison now returns true if data is approximately equal in the range of a precision value 
- Fixed bug in homogeneity calculation that resulted in wrong values when the sampling interval is not equal in both
  axes.
- Added OS3D file format
- Fixed issues with cython imports during building of documentation
- Replaced default roughness parameters calculated by Surface.roughness_parameters() by ISO parameters, omitting the 
  custom parameters
- Added filename parsing to batch processing to conveniently extract parameters from filenames
- Added tests
- Expanded docs
## v0.9.2
- Fixed bug with type hinting
## v0.9.1
- Fixed missing reading of image layers with SUR files
- Added tests for reading of image layers
## v0.9.0
- Fixed bug with incorrect unit conversion of OPD files
- Moved height parameter calculation to Cython, reducing processing time by a factor of 100
- Refactored Sdr calculation and removed Gwyddion surface area algorithm
- Moved package version to single file, all other occurences of the version number now read that file. The version
  is now available with the __version__ attribute of the package
- Major overhaul of the file readers:
  - Support for reading of image layers (Grayscale, RGB, Intensity)
  - Support for plotting of image layers instead of the topography layer in the Surface class
  - Added Image class as a thin wrapper around image arrays to facilitate saving to disk
  - Support for extraction of metadata from the files. Metadata is now availbale as a dictionary in the Surface class
  - Support for the newest version of the sur file standard, including compressed sur files
  - Support for Gwyddion file format
  - The following fileformats now support image reading: vk4, vk6, vk7, plu, plux, sur, opd, nms, gwy
- Removed documentation from the readme file. Docs are now exlusively found on readthedocs
- Added some documentation for interacting with image layers
## v0.8.2
- Cython surface area calculation now releases the GIL, which is necessary to parallelize the computation with the new 
  thread pool based batch execution
- Fixed bug that causes overflow of the height values while reading SUR files with specific unit conversion factors
- Fixed incorrect calculation of standard deviation for Gaussian filter (thanks to Dorothee Hüser for the correction)
## v0.8.1
- Added kwarg to Gaussian filter to define endeffect management method
- Sinusoid now raised FittingError when fitting fails
- Batch.execute now uses threadpool instead of multiprocessing to avoid issues with jupyter and pickling, while
  maintaining almost the same speed gains due to frequent GIL release in numpy-based computations
## v0.8.0
- Added Alicona .al3d file format 
- Fixed bug resulting in negative Vmc and Vmp
- Added Surface.save method to export a surface to different file formats. So far, .sur and .al3d are supported
- Added .sdf file format
- Added test files for all supported file formats and unittests for loading the testfiles
- Switched from scipy-based calculation of autocorrelation function to fft-based implementation, which corresponds to
  the approach that MountainsMap seems to be using
- Autocorrelation now only centers the surface and does not level it anymore, to ensure correspondence with MountainsMap
- Setup.py now autodetects and compiles all Cython modules 
## v0.7.0
- Surface.zero and Surface.level fixed for surfaces with nonmeasured points
- Added support for Nanofocus NMS file format
- Added encoding option to Surface.load as keyword argument 
## v0.6.0
- Added Zeta .zmg file format
- Homogeneity now raises value error if any periodic or negative parameter is specified as input
- Period can now be manually set for homogeneity calculation by keyword argument
- Fixed multiprocessing error on Windows that caused infinite spawning of child processes
- Added openpyxl as dependency
- Added from_dir constructor classmethod to Batch, to initialize batch object from a directory path that contains the
  topography files as a convenience function
- Fixed bug with loading Gwyddion exported SUR files due to Gwyddion filling strings with null instead of spaces
- Added OPD file format (OPD files may have differing values of step_x and step_y, this could cause unexpected errors 
  in the calculation of some parameters)
- Added XYZ file format. The reader assumes units of meters and data points on a regular grid.
- Added methods to level and calculate stepheight and cavity volume for rectangular ablation craters
## v0.5.1
- Changed plot parameter of Surface.depth to plot a specific or multiple profiles
- Fixed hashing of mutable types for caching of method calls with mutable parameters
- Added sphinx documentation and readthedocs page
- Added Sinusoid class that can be constructed from parameters or from fitting and implements a method for calculation
  of the position of the first extremum
- Depth calculation now starts from first extremum, which now makes sure to be independent of possibly large fit values
  of x0
- Added small offset to fourier transform for plotting if log=True to avoid log(0) error
- Fixed and simplyfied the calculation of the DFT peaks
- Added cropping to Batch
## v0.5.0
- Added algorithm for computing texture orientation with significantly higher precision than the current fft-based 
  method. The purely fft-based method can still be selected via keyword argument.
- Replaced functools.lru_cache with custom cache implementation since lru_cache causes memory leaks when applied to 
  instance methods. This would result in a substantial memory leak during batch processing because the surface objects
  would not be garbage collected once they went out of scope. 
- Added cache to the height paramters, reducing calculation time by ~20% since Sq is invoked multiple times. 
- Fixed dtype mismatch in Cython code for topographies with float32 instead of float64 dtype (usually plu and plux 
  files) 
- Leveling now works on surfaces with non-measured points
## v0.4.0:
- Reduced file load times by >90% using numpy.fromfile to read the height data
- Leveling now no longer also centers the data around the mean
- Fixed wrong alignment direction in Surface.align
- Reverted to binning with default of 10,000 bins for AbbottFirestoneCurve due to large performance bottleneck of
  functional parameter calculations based on whole dataset
- Added support for multiple return values in the Batch class
- Surface.align now takes axis argument to specify with which axis to align the texture
- Surface.depth now considers orientation of the structure and uses correct period in x or y depending on it. It also
  now uses either horizontal or vertical profiles depending on the orientation. Moreover, it doesn't compute the period
  initial guess from each profile anymore but only once from the fourier transform.
- Reworked file loading code and added support for .sur files
- Added basic operator overloading for arithmetic operations to Surface, e.g. "surface + 5.2" or "surface1 - surface2"
- Added Gaussian filter and removed FFT-filter. Surface.filter now invokes a Gaussian filter. 
## v0.3.0:
- Removed underscore from Surface._data, ._step_x, ._step_y, ._width_um, ._height_um since they are supposed to be
  public attributes
- Fixed bug in Surface.zoom
- Reworked Batch processing: now implements lazy loading and dispatch to multiple processes for faster load time
- Operations and Parameters now are constructured as a pipeline on the Batch object, with the actual execution only
  taking place once Batch.execute is called
- Dataframe is now constructed after all processing is completed, which reduces complexity
- Added keyword argument for saving dataframe to excel upon completion of Batch.execute
- Additional data is now loaded directly when initializing Batch object and raises ValueError if column with name file
  doesn't exist
- Updated readme with examples for batch processing
- Added cropping method to Surface
- Fixed bug in plotting of depth analysis
- Added thresholding method based on material ratio to Surface
- Added outlier removal method based on sigma criterion
- Added support for operation on data which contains non-measured points to thresholding and outlier removal methods
- Added remove_outliers and threshold to Batch
- Removed binning from AbbottFirestoneCurve. Instead, now calculates the curve from all available points with unequal
  spacing.
- Fixed errors in homogeneity calculation and added parameter to specify roughness parameters for evaluation on method
  call.
- tqdm is now a non-optional dependency
## v0.2.0:
- Added Sdq parameter -> tested against LeicaMap
- Removed width_um and height_um as initialization parameters for Surface, since they are redundant
- Added lru_cache to some functions to save computation time
- Added cache clearing mechanism as well as recalculation of width_um and height_um upon inplace modification of data
- Replaced default method for calculation of surface area with the one proposed by ISO 25178-2. The method used by
  Gwyddion is now available by keyword argument. Sdr values now match the values computed by MountainsMap and its
  derivative software packages.
- Added correct tests for ISO Sdr values
- Added Smr(c), Smc(mr) and Sxp to the functional parameters
- Moved functional parameter calculation to new class AbbottFirestoneCurve
- Added functional volume parameters Vmp, Vmc, Vvv, Vvc
- Added tests for functional volume parameters
- Added Fourier transform plotting method to Surface
- Fixed a bug with surface area calculation in the Cython code that would result in negative Sdr for some edgecases
- Added autocorrelation function and the derived spatial parameters Sal and Str
- Added tests for spatial parameters
- Restructuring of modules: Profile, AbbottFirestoneCurve, AutocorrelationFunction now in speparate modules
- Fixed filtering method: Now filters at the correct frequencies in all axes. Added note in docstring that warns about
  the side effects of zeroing bins
## v0.1.0:
- Added option to Batch to supply additional parameters associated with each file that can be merged into the
  computed parameters
- Fixed bug where Surface.rotate() would leave a border of nan values
- Fixed bug in profile averaging
- Fixed calculation of Surface._get_fourier_peak_dx_dy(), now returns correct angles with correct sign
- Fixed Surface.orientation(), now considers edgecases where either dy or dx are 0
- Added calculation of areal material ratio curve
- Fixed Abbott-Firestone curve display
- Added calculation fundament for functional parameters
- Added some functional parameters: Sk, Spk, Svk, Smr1, Smr2
- Added utils module with some helper functions
- Fixed bug in depth calculation due to renaming of sinusoid function
- Added aspect ratio parameter to Surface
- Added tests for roughness parameters
- Added script to generate tests for roughness parameters for a set of example surfaces
