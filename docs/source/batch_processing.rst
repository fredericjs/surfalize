================
Batch processing
================

Setting up the batch
====================

surfalize provides a module for batch processing and surface roughness evaluation of large sets of measurement files.
The `Batch` object is created by supplying a list or generator object of filepaths to the topography files.

.. code:: python

    from pathlib import Path
    from surfalize import Batch

    filepaths = Path('folder').glob('*.vk4')

    # Create a Batch object that holds the filepaths to the surface files
    batch = Batch(filepaths)

Alternatively, the `Batch` class provides an alternative constructor to initialize the a `Batch` directly from a folder
containing topography files. If the `extension` argument is not defined, all files corresponding to supported files
formats will be loaded. Alternatively, a list of specific formats can also be supplied.

.. code:: python

    batch = Batch.from_dir('path/to/folder/', extension='.vk4')



All operations of the surface can be applied to the Batch analogously to a Surface object.
However, they are not applied immediately but registered for later execution.

.. code:: python

    batch.level()
    batch.filter('highpass', 20)


Each operation on the batch returns the Batch object, allowing for method chaining.

.. code:: python

    batch = Batch(filepaths).level().filter('highpass', 20).align().center()

The calculation of roughness parameters can be done indiviually and chained.

.. code:: python

    batch.Sa().Sq().Sq().Sdr()

Arguments to the roughness parameter calculations, such as _p_ and _q_ can be provided in the individual call.

.. code:: python

    batch.Vmc(p=10, q=80)

Parameters can also be calculated in bulk using `Batch.roughness_parameters()`:

.. code:: python

    # Computes Sa, Sq, Sz
    batch.roughness_parameters(['Sa', 'Sq', 'Sz'])
    # Computes all available parameters
    batch.roughness_parameters()

If arguments need to be supplied, the parameter must be constructed as a `Parameter` object:

.. code:: python

    from surfalize.batch import Parameter
    Vmc = Parameter('Vmc', kwargs=dict(p=10, q=80))
    batch.roughness_parameters(['Sa', 'Sq', 'Sz', Vmc])


Finally, the batch processing is executed by calling `Batch.execute`, returning a `pd.DataFrame`. Optionally,
`multiprocessing=True` can be specified to split the load among all available CPU cores. Moreover, the results
can be saved to an Excel Spread sheet by specifiying a path for `saveto=r'path\to\excel_file.xlsx`.

.. code:: python

    df = batch.execute(multiprocessing=True)

Optionally, a Batch object can be initialized with a filepath pointing to an Excel File which contains additional
parameters, such as laser parameters. The file must contain a column `file`, which specifies the filename including file
extension in the form `name.ext`, e.g. `topography_50X.vk4`. All other columns will be merged into the resulting
Dataframe that is returned by `Batch.execute`.

.. code:: python

    batch = Batch(filespaths, additional_data=r'C:\users\exampleuser\documents\laserparameters.xlsx')
    batch.level().filter('highpass', 20).align().roughness_parameters()
    df = batch.execute()

Full example
============

Let's supppose we have four topography files called `topo1.vk4`, `topo2.vk4`, `topo3.vk4`, `topo4.vk4` in
the folder `C:\users\exampleuser\documents\topo_files`. Moreover, we have additional information on these files in an
Excel files located in `C:\users\exampleuser\documents\topo_files\laserparameters.xlsx`. The Excel looks like this:


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
    batch.execute(multiprocessing=True, saveto=r'C:\users\exampleuser\documents\roughness_results.xlsx')

The result will be a DataFrame that looks like this:

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
