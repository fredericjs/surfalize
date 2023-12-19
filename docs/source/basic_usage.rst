.. default-role:: code

===========
Basic usage
===========

.. code:: python

    from surfalize import Surface

    filepath = 'example.vk4'
    surface = Surface.load(filepath)
    surface.show()


Extracting roughness and topographic parameters
===============================================

.. code:: python

    # Individual calculation of parameters
    sa = surface.Sa()
    sq = surface.Sq()

    # Calculation in batch
    parameters = surace.roughness_parameters(['Sa', 'Sq', 'Sz'])
    >>> parameters
    {'Sa': 1.25, 'Sq': 1.47, 'Sz': 2.01}

    # Calculation in batch of all available parameters
    all_available_parameters = surace.roughness_parameters()


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
    surface = surace.remove_outliers()

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
    frequencies to fxmax and fymax.
    surface.plot_fourier_transform(hanning=True, fxmax=2, fymax=1)


