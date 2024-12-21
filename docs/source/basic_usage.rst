.. default-role:: code

===========
Basic usage
===========

All 2d topographies in surfalize are represented by instances of the `Surface` class. To instantiate a surface object
from a file, simply use its `.load` classmethod and pass it a filepath. Use the `.show()` method to plot a colorcoded
topographical representation of it's data. In Jupyter Notebooks, the surface object can simply be stated in the last
line of a cell to invoke its repr method, displaying the topography without the need for calling the `.show()` method.

.. code:: python

    from surfalize import Surface

    filepath = 'example.vk4'
    surface = Surface.load(filepath)
    surface.show()

Surface objects can also be instantiated from file-like objects that live in memory. This can be useful for instance
when the objects are obtained directly from a database connection. In this example, we read the file from disk into a
buffer allocated by an `io.BytesIO` object. We then pass the buffer the load `.load()` method instead of a filepath.

.. code:: python

    import io
    from surfalize import Surface

    filepath = 'example.vk4'
    with open(filepath, 'rb') as f:
        buffer = io.BytesIO(f.read())
    surface = Surface.load(buffer)
    surface.show()

If the filepath has a suffix (as it should), surfalize determines the file format from the filepath suffix. If, however,
the file path has no suffix, if a buffer object is passed instead of a filepath or the filepath has a wrong suffix and
the file reading operation raises an error, surfalize tries to infer the correct file format from the file magic.
For reading from file-like objects or for overriding the fileformat determined by the file extension, the `format`
argument of the `load()` method can be specified:

.. code:: python

    surface = Surface.load(buffer, format='.vk4')




Extracting roughness and topographic parameters
===============================================

All roughness parameters can be calculated via methods of the `Surface` class.
The methods are named analogous to the parameters defined in the ISO 25178 standard for the 
standardized parameters with a capitalized first letter.

.. code:: python

    # Individual calculation of parameters
    sa = surface.Sa()
    sq = surface.Sq()

.. code:: python

    # Calculation in batch
    parameters = surface.roughness_parameters(['Sa', 'Sq', 'Sz'])
    >>> parameters
    {'Sa': 1.25, 'Sq': 1.47, 'Sz': 2.01}

    # Calculation in batch of all available parameters
    all_available_parameters = surface.roughness_parameters()


Performing surface operations
=============================

Surface operations return a new `Surface` object by default. If `inplace=True` is specified, the operation applies
to the `Surface` object it is called on and returns it to allow for method chaining. Inplace is generally faster since
it does not copy the data and does not need to instantiate a new object.

.. code:: python

    # Levelling the surface using least-sqaures plane compensation

    # Returns new Surface object
    surface = surface.level()

    # Returns self (to allow for method chaining)
    surface.level(inplace=True)

    # Filters the surface using a Gaussian filter
    surface_low = surface.filter(filter_type='highpass', cutoff=10)

    # Filter form and noise
    surface_filtered = surface.filter(filter_type='bandpass', cutoff=0.8, cutoff2=10)

    # Separate waviness and roughness at a cutoff wavelength of 10 µm and return both
    surface_roughness, surface_waviness = surface.filter('both', 10)

    # If the surface contains any non-measured points, the points must be interpolated before any other operation can be applied
    surface = surface.fill_nonmeasured(method='nearest')

    # The surface can be rotated by a specified angle in degrees
    # The resulting surface will automatically be cropped to not contain any areas without data
    surface = surface.rotate(10)

    # Aligning the surface texture to a specified axis by rotation, default is y
    surface = surface.align(axis='y')

    # Remove outliers
    surface = surface.remove_outliers()

    # Threshold the surface, default threshold value is 0.5% of the AbbottCurve
    surface = surface.threshold(threshold=0.5)
    # Non-symmetrical tresholds can be specified
    surface = surface.threshold(threshold=(0.2, 0.5))

These methods can be chained:

.. code:: python

    surface = Surface.load(filepath).level().filter(filter_type='lowpass', cutoff=0.8)
    surface.show()


Plotting
========

The `Surface` object offers multiple types of plots.

Plotting the topography itself is done using `Surface.show()`. If the repr of a `Surface` object is
invoked by Jupyter Notebook, it will automaticall call `Surface.show()`.

.. code:: python

    # Some arguments can be specified
    surface.show(cmap='viridis', maskcolor='red')

The Abbott-Firestone curve and Fourier Transform can be plotted using:

.. code:: python

    surface.plot_abbott_curve()
    # Here we apply a Hanning window to mitigate spectral leakage (recommended) as crop the plotted range of
    # frequencies to fxmax and fymax.
    surface.plot_fourier_transform(hanning=True, fxmax=2, fymax=1)

Accessing the raw data
======================

The raw data of a `Surface` object can be accessed with the attribute `data` as a two-dimensional `numpy` array.
The pixel resolution in x (horizontal) and y (vertical) is accessed through the attributes `step_x` and `step_y`.
The width and height in micrometers are accessed through the attributed `width_um` and `height_um`. The resolution in
pixels is encoded in the named tuple `size`, holding the dimensions in the form `(y, x)`.


.. code:: python

    data_2d = surface.data
    step_x = surface.step_x
    step_y = surface.step_y
    ny, nx = surface.size
    # or:
    nx = surface.size.x
    ny = surface.size.y
    width = surface.width_um
    height = surface.height_um

Working with image and metadata
===============================

Surfalize can read image data and metadata from several file formats. The metadata can be accessed through

.. code:: python

    surface = Surface.load('path.ext')
    metadata = surface.metadata
    print(metadata)

    >>> {'Time': 'DD/MM/YYYY', 'Objective': '50X', ...}

Optionally, image layers, such as RGB, intensity or Grayscale image present in the file can be read by specifying
`read_image_layers=True`, in `Surface.load`. The image layers can then be accessed in the dictionary
`Surface.image_layers`. Grayscale and RGB images generally have the keys 'Grayscale' and 'RGB', respectively, if no
other title is specified in the file or file format specification. The images are represented by an `Image` class,
which is a thin wrapper around a numpy array, that provides additional functionality for saving the image to disk.

.. code:: python

    surface = Surface.load('path.ext', read_image_layers=True)
    print(surface.image_layers)

    >>> {'RGB': Image(736 x 480, Bitdepth: 8, Mode: RGB), 'Intensity': Image(736 x 480, Bitdepth: 16, Mode: Grayscale)}

Image layers can be saved to disk by calling their `.save` method. The raw image data can be accessed in the Image's
`data` attribute.

.. code:: python

    surface.image_layers['RGB'].save('C:/image.png') # save the image
    raw_data = surface.image_layers['RGB'].data # returns numpy array

The `Surface.show` method can be used to plot image layers instead of the topography layer.

.. code:: python

    surface.show(layer='RGB')
