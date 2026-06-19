.. default-role:: code

===========
Basic usage
===========

All 2d topographies in surfalize are represented by instances of the :code:`Surface` class. To instantiate a surface object
from a file, simply use its :code:`.load` classmethod and pass it a filepath. Use the :code:`.show()` method to plot a colorcoded
topographical representation of it's data. In Jupyter Notebooks, the surface object can simply be stated in the last
line of a cell to invoke its repr method, displaying the topography without the need for calling the :code:`.show()` method.

.. code:: python

    from surfalize import Surface

    filepath = 'example.vk4'
    surface = Surface.load(filepath)
    surface.show()

Surface objects can also be instantiated from file-like objects that live in memory. This can be useful for instance
when the objects are obtained directly from a database connection. In this example, we read the file from disk into a
buffer allocated by an :code:`io.BytesIO` object. We then pass the buffer the load :code:`.load()` method instead of a filepath.

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
For reading from file-like objects or for overriding the fileformat determined by the file extension, the :code:`format`
argument of the :code:`load()` method can be specified:

.. code:: python

    surface = Surface.load(buffer, format='.vk4')




Extracting roughness and topographic parameters
===============================================

All roughness parameters can be calculated via methods of the :code:`Surface` class.
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

Surface operations return a new :code:`Surface` object by default. If :code:`inplace=True` is specified, the operation applies
to the :code:`Surface` object it is called on and returns it to allow for method chaining. Inplace is generally faster since
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


Masking regions
===============

Regions of a surface can be masked so that they are excluded from analysis while their height values are preserved.
This is useful to keep artifacts, dust particles or otherwise invalid areas from influencing operations such as
levelling or the calculation of height parameters. Masking is reversible, in contrast to non-measured points, which are
stored as :code:`NaN` and irreversibly discard the height value.

Every :code:`Surface` owns a :code:`Mask` object, accessible through the :code:`surface.mask` attribute. The mask is
empty until a region is masked. Masked points are excluded from the height parameters (:code:`Sa`, :code:`Sq`,
:code:`Sk`, ...), the statistical reductions (:code:`min`, :code:`max`, :code:`mean`, ...) and the levelling fit. The
convention follows :code:`numpy.ma`, i.e. a value of :code:`True` marks a point that is masked (ignored).

.. note::

    Parameters that require a complete, gap-free grid (e.g. :code:`Sdr`, :code:`Sdq`, :code:`Sal`, :code:`Str`,
    :code:`period`, :code:`orientation`) are not defined on a masked surface and will raise an error. Clear the mask
    before computing them.

Consistent with the other surface operations, the masking methods return a new :code:`Surface` by default and leave
the original unchanged. Specify :code:`inplace=True` to modify the surface in place and return it, which also allows
chaining.

.. code:: python

    # Returns a new Surface with the masked region, leaving the original untouched
    masked = surface.mask.add_rectangle((0, 10, 0, 10))

    # Modify the surface in place
    surface.mask.add_circle((50, 50), radius=5, inplace=True)

Several regions can be added or removed. The region methods interpret their coordinates in physical units (µm) by
default, following the same convention as :code:`Surface.crop` where the y-axis is measured from the bottom. Passing
:code:`in_units=False` switches to pixel indices.

.. code:: python

    # Mask by geometry (µm by default)
    surface = surface.mask.add_rectangle((0, 10, 0, 10))     # (x0, x1, y0, y1)
    surface = surface.mask.add_circle((50, 50), radius=5)
    surface = surface.mask.subtract_circle((50, 50), radius=2)

    # Mask by height value
    surface = surface.mask.threshold(below=-1, above=5)

    # Invert or clear the mask
    surface = surface.mask.invert()
    surface = surface.mask.clear()

In addition to the region methods, the mask supports numpy-style indexing in pixel coordinates. Item assignment is
always performed in place, since Python item assignment cannot return a value.

.. code:: python

    surface.mask[10:20, 5:15] = True      # mask a slice
    surface.mask[surface.data > 5] = True  # mask by a boolean condition

    has_mask = surface.has_masked_points   # True if any point is masked
    mask_array = surface.mask.to_array()   # boolean array of the masked points

The mask travels with the data through slicing and the :code:`crop`, :code:`zoom` and :code:`__getitem__` operations,
and is preserved by levelling. A deep copy of a surface, including its mask, can be obtained with :code:`surface.copy()`.


Plotting
========

The :code:`Surface` object offers multiple types of plots.

Plotting the topography itself is done using :code:`Surface.show()`. If the repr of a :code:`Surface` object is
invoked by Jupyter Notebook, it will automaticall call :code:`Surface.show()`.

.. code:: python

    # Some arguments can be specified
    # maskcolor sets the color of non-measured points
    surface.show(cmap='viridis', maskcolor='black')

Masked points are drawn as a translucent overlay on top of the topography so that the underlying height still shows
through. Its color and opacity can be adjusted with the :code:`masked_color` and :code:`masked_alpha` arguments
(defaults are :code:`'red'` and :code:`0.5`).

.. code:: python

    surface.show(masked_color='orange', masked_alpha=0.3)

The Abbott-Firestone curve and Fourier Transform can be plotted using:

.. code:: python

    surface.plot_abbott_curve()
    # Here we apply a Hanning window to mitigate spectral leakage (recommended) as crop the plotted range of
    # frequencies to fxmax and fymax.
    surface.plot_fourier_transform(hanning=True, fxmax=2, fymax=1)

Accessing the raw data
======================

The raw data of a :code:`Surface` object can be accessed with the attribute :code:`data` as a two-dimensional :code:`numpy` array.
The pixel resolution in x (horizontal) and y (vertical) is accessed through the attributes :code:`step_x` and :code:`step_y`.
The width and height in micrometers are accessed through the attributed :code:`width_um` and :code:`height_um`. The resolution in
pixels is encoded in the named tuple :code:`size`, holding the dimensions in the form :code:`(y, x)`.


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
:code:`read_image_layers=True`, in :code:`Surface.load`. The image layers can then be accessed in the dictionary
:code:`Surface.image_layers`. Grayscale and RGB images generally have the keys 'Grayscale' and 'RGB', respectively, if no
other title is specified in the file or file format specification. The images are represented by an :code:`Image` class,
which is a thin wrapper around a numpy array, that provides additional functionality for saving the image to disk.

.. code:: python

    surface = Surface.load('path.ext', read_image_layers=True)
    print(surface.image_layers)

    >>> {'RGB': Image(736 x 480, Bitdepth: 8, Mode: RGB), 'Intensity': Image(736 x 480, Bitdepth: 16, Mode: Grayscale)}

Image layers can be saved to disk by calling their :code:`.save` method. The raw image data can be accessed in the Image's
:code:`data` attribute.

.. code:: python

    surface.image_layers['RGB'].save('C:/image.png') # save the image
    raw_data = surface.image_layers['RGB'].data # returns numpy array

The :code:`Surface.show` method can be used to plot image layers instead of the topography layer.

.. code:: python

    surface.show(layer='RGB')
