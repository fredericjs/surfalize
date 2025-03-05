================
Batch processing
================

Setting up the batch
====================

surfalize provides a module for batch processing and surface roughness evaluation of large sets of measurement files.
The :code:`Batch` object is created by supplying a list or generator object of filepaths to the topography files.

.. code:: python

    from pathlib import Path
    from surfalize import Batch

    filepaths = Path('folder').glob('*.vk4')

    # Create a Batch object that holds the filepaths to the surface files
    batch = Batch(filepaths)

Alternatively, the :code:`Batch` class provides an alternative constructor to initialize the a :code:`Batch` directly
from a folder containing topography files. If the :code:`extension` argument is not defined, all files corresponding to
supported files formats will be loaded. Alternatively, a list of specific formats can also be supplied.

.. code:: python

    batch = Batch.from_dir('path/to/folder/', extension='.vk4')

To pass file-like objects to a Batch object, they must first be wrapped in an instance of the :code:`FileInput` class to
provide a name and optionally a file format specifier.

.. code:: python

    import io
    from surfalize import Batch, FileInput

    # Here we create a file-like object for the sake of demonstration. In practice, these probably come from a database
    # or network connections

    with open('example_1.vk4', 'rb') as f:
        buffer1 = io.BytesIO(f.read())

    with open('example_2.vk4', 'rb') as f:
        buffer2 = io.BytesIO(f.read())

    fileobj1 = FileInput(name='my_surface_1', data=buffer, format='.vk4')
    fileobj2 = FileInput(name='my_surface_2', data=buffer, format='.vk4')

    batch = Batch([fileobj1, fileobj2])

Applying operations
===================

All operations of the surface can be applied to the Batch analogously to a Surface object. The batch object essentially
acts as an almost drop-in replacement for the surface object. However, operations and calculations are not applied
immediately but registered for later execution.

.. code:: python

    batch.level()
    batch.filter('highpass', 20)


Each operation on the batch returns the Batch object itself, allowing for method chaining.

.. code:: python

    batch = Batch(filepaths).level().filter('highpass', 20).align().center()

Calculating parameters
======================

The calculation of roughness parameters can be done indiviually and chained.

.. code:: python

    batch.Sa().Sq().Sq().Sdr()

Arguments to the roughness parameter calculations, such as :code:`p` and :code:`q` can be provided in the individual
call.

.. code:: python

    batch.Vmc(p=10, q=80)

Parameters can also be calculated in bulk using :code:`Batch.roughness_parameters()`:

.. code:: python

    # Computes Sa, Sq, Sz
    batch.roughness_parameters(['Sa', 'Sq', 'Sz'])
    # Computes all available parameters
    batch.roughness_parameters()

If arguments need to be supplied, the parameter must be constructed as a :code:`Parameter` object:

.. code:: python

    from surfalize.batch import Parameter
    Vmc = Parameter('Vmc', kwargs=dict(p=10, q=80))
    batch.roughness_parameters(['Sa', 'Sq', 'Sz', Vmc])

Execution order
===============

Before version :code:`v0.15.0` all operations were executed before parameter calculations. For versions
:code:`>=v0.15.0`,
Operations and parameters can be called in an interlaced manner and their order will be executed in that order. This
allows for cases where the user wants to calculate some parameters before and others after a specific operation.
The legacy behavior of performing all operations first can be activated by specifying
:code:`presever_chaining_order=False` in :code:`Batch.execute`.

In this example, :code:`Sdr` will be calculated before the filtering and :code:`Sq` after the filtering:

.. code:: python

    batch = Batch.from_dir('.')
    batch.Sdr().filter('lowpass', 1).Sq()
    result = batch.execute()

In this example, :code:`Sdr` and :code:`Sq` will be calculated after the filtering:

.. code:: python

    batch = Batch.from_dir('.')
    batch.Sdr().filter('lowpass', 1).Sq()
    result = batch.execute(preserve_chaining_order=False)

Duplicate Parameters
====================

In some cases, one might want to calculate the same parameter multiple times, for instance before and after an operation
or with different arguments. If a parameter is called more than once on the :code:`Batch` object, an exception is raised
to prevent the column in the dataframe being overwritten by the second call. However, each parameter can be given a
custom name for its column in the dataframe to enable duplicate calculation of the same parameter:

In this example, we calculate :code:`Sdr` before and after filtering the surface with a highpass filter to investigate,
how strongly the high frequency noise affects the parameter's value:

.. code:: python

    batch = Batch.from_dir('.')
    batch.Sdr().filter('lowpass', 1).Sdr(custom_name='Sdr_after_filtering')
    result = batch.execute()

In this example, we calculate the homogeneity with different unit cell evaluation parameters:

.. code:: python

    batch = Batch.from_dir('.')
    batch.homogeneity(parameters=['Sa'], custom_name='H_Sa')
    batch.homogeneity(parameters=['Sa', 'Sk', 'Sdr'], custom_name='H_Sa_Sk_Sdr')
    result = batch.execute()


Executing the batch process
===========================

Finally, the batch processing is executed by calling :code:`Batch.execute`, returning a :code:`BatchResult` object. The
:code:`BatchResult` class wraps a :code:`pd.DataFrame` object (but is not a subclass of it) and exposes all its methods.
Therefore, it can be used like a :code:`DataFrame` for most purposes but also offers some additional functionality. To
access the underlying :code:`DataFrame` object, the method :code:`get_dataframe` can be called on the object.
Optionally, :code:`multiprocessing=True` can be specified to :code:`Batch.execute` to split the load among all available
CPU cores. Moreover, the results can be saved to an Excel Spread sheet by specifiying a path for
:code:`saveto=r'path\to\excel_file.xlsx`.

.. code:: python

    result = batch.execute(multiprocessing=True)

If the calculation of one parameter fails for even one surface, which could be the case for instance when a
:code:`FittingError` occurs during the calculation of the structure depth, the entire batch processing stops and the error
is raised. This is often unwanted behavior, when a large dataset is batch processed. To avoid this, surfalize ignores
errors that occur during batch processing and fills the parameters that raised an error during calculation with :code:`NaN`
values. If you specifically want any errors to be raised nonetheless, specify :code:`ignore_errors=False`.

.. code:: python

    result = batch.execute(multiprocessing=True, ignore_errors=False)

Optionally, a Batch object can be initialized with a filepath pointing to an Excel File which contains additional
parameters, such as laser parameters. The file must contain a column :code:`file`, which specifies the filename including file
extension in the form :code:`name.ext`, e.g. :code:`topography_50X.vk4`. All other columns will be merged into the resulting
Dataframe that is returned by :code:`Batch.execute`.

.. code:: python

    batch = Batch(filespaths, additional_data=r'C:\users\exampleuser\documents\laserparameters.xlsx')
    batch.level().filter('highpass', 20).align().roughness_parameters()
    result = batch.execute()

Parsing filenames for additional parameters
===========================================
Oftentimes, the filenames of the topography files encode parameters that are in some way associated with the measured
topography. For instance, one might encode the fabrication parameters in the filename, following a specific layout.
In order to extract these parameters from the filenames into individual columns in the dataframe, the use must spend
some time, for instance to construct a working regex, parse the filenames, convert the resulting columns to the
respective types and so on.

To streamline this process, surfalize offers a convenient way to define a filename format, from which the parameters
can be extracted. For instance, a surface might be fabricated by a laser process using the following parameters:

* Fluence: 1.21 J/cm²
* Frequency: 100 kHz
* Scanspeed: 1 m/s
* Hatch distance: 100 µm
* Overscans: 5

The filename might encode these values in the following way:

:Filename: `F1.21_FREP100kHz_V1_HD100_OS5.vk6`

To parse this filename, you can define a template string, where each parameter is specified in angular brackets by
specifying their name, datatype, prefix (optional) and suffix (optional). The name is used to label the resulting
column in the dataframe. The patterns have the general syntax:

:Template syntax: `<name|datatype|prefix|suffix>`

Both prefix and suffix can be omitted. If only a suffix is defined, the prefix must be indicated as an empty string.
The exemplary filename could be parsed in using the following template string:

:Template string: `<fluence|float|F>_<frequency|float|FREP|kHz>_<scanspeed|float|V>_<hatch_distance|float|HD>_<overscans|int|OS>`

The possible datatypes that can be matched are str, int, float.

To apply the filename extraction based on the defined template string, you can call the respective method on the batch
object:

.. code:: python

    batch = Batch.from_dir('.')
    batch.level()
    pattern = '<fluence|float|F>_<frequency|float|FREP|kHz>_<scanspeed|float|V>_<hatch_distance|float|HD>_<overscans|int|OS>'
    batch.extract_from_filename(pattern)
    batch.roughness_parameters()
    result = batch.execute()

Instead of on the `Batch` object, the filename extraction can also be applied on the `BatchResult` object, which has the
advantage that the Batch does not have to be executed every time the template string is changed, for instance when the
template string was constructed wrong. The method `BatchResult.extract_from_filename` operates inplace on the object.

.. code:: python

    batch = Batch.from_dir('.')
    batch.level()
    batch.roughness_parameters()
    result = batch.execute()
    pattern = '<fluence|float|F>_<frequency|float|FREP|kHz>_<scanspeed|float|V>_<hatch_distance|float|HD>_<overscans|int|OS>'
    result.extract_from_filename(pattern)

Adding custom parameters
========================

Custom parameters can be added to the batch calculation by passing a user defined function to :code:`Batch.custom_parameter`.
This function must take only one argument, which is the surface object. It must return a dictionary, where the key
represents the name of the parameter that is used for the column name in the DataFrame and the value is the result of
the calculation. If multiple return values are needed, each must be inserted with a different key into the dictionary.

.. code:: python

    # With one return value
    def median(surface):
        median = np.median(surface.data)
        return {'height_median': median}

    # With multiple return values
    def mean_std(surface):
        mean = np.mean(surface.data)
        std = np.std(surface.data)
        return {'mean_value': mean, 'std_value': std}

    # Register the functions for batch execution
    batch.custom_parameter(median)
    batch.custom_parameter(mean_std)

Full example
============

Let's supppose we have four topography files called :code:`topo1.vk4`, :code:`topo2.vk4`, :code:`topo3.vk4`, :code:`topo4.vk4` in
the folder :code:`C:\users\exampleuser\documents\topo_files`. Moreover, we have additional information on these files in an
Excel files located in :code:`C:\users\exampleuser\documents\topo_files\laserparameters.xlsx`. The Excel looks like this:


+------------+-------+---------------+----------------+
| file       | power | pulse_overlap | hatch_distance |
+============+=======+===============+================+
| topo1.vk4  | 100   | 20            | 12.5           |
+------------+-------+---------------+----------------+
| topo2.vk4  | 50    | 20            | 12.5           |
+------------+-------+---------------+----------------+
| topo3.vk4  | 100   | 50            | 12.5           |
+------------+-------+---------------+----------------+
| topo4.vk4  | 50    | 50            | 12.5           |
+------------+-------+---------------+----------------+

.. code:: python

    from pathlib import Path
    from surfalize import Batch

    filepaths = Path(r'C:\users\exampleuser\documents\topo_files').glob('*.vk4')
    batch = Batch(filespaths, additional_data=r'C:\users\exampleuser\documents\topo_files\laserparameters.xlsx')
    batch.level().filter('highpass', 20).align().roughness_parameters(['Sa', 'Sq', 'Sz'])
    result = batch.execute(multiprocessing=True, saveto=r'C:\users\exampleuser\documents\roughness_results.xlsx')

The result will be a BatchResult that looks like this:

+------------+-------+---------------+----------------+------+------+------+
| file       | power | pulse_overlap | hatch_distance | Sa   | Sq   | Sz   |
+============+=======+===============+================+======+======+======+
| topo1.vk4  | 100   | 20            | 12.5           | 0.85 | 1.25 | 3.10 |
+------------+-------+---------------+----------------+------+------+------+
| topo2.vk4  | 50    | 20            | 12.5           | 0.42 | 0.51 | 1.87 |
+------------+-------+---------------+----------------+------+------+------+
| topo3.vk4  | 100   | 50            | 12.5           | 1.34 | 1.67 | 3.84 |
+------------+-------+---------------+----------------+------+------+------+
| topo4.vk4  | 50    | 50            | 12.5           | 0.55 | 0.67 | 1.99 |
+------------+-------+---------------+----------------+------+------+------+
