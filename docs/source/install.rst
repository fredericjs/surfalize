============
Installation
============

To install :code:`surfalize` from PyPI run:

.. code::

    pip install surfalize

Surfalize offers optional dependencies for various functions, such as 3d plotting (3d), command line functionality (cli),
running tests (tests) building the documentation (docs). The install, put the name of the optional dependency set in
square brackets behind the package name. For instance, optional dependencies for 3d plotting can be installed like this:

.. code::

    pip install surfalize[3d]

Multiple optional dependencies can be specified with a comma:

.. code::

    pip install surfalize[3d,cli]

Or all optional dependencies can be installed using:

.. code::

    pip install surfalize[all]

Alternatively, you can clone the git repository, navigate to the root folder and run:

.. code::

    pip install .


You can also install directly from git via pip:

.. code::

    pip install git+https://github.com/fredericjs/surfalize.git

Optionally, you can specify the branch to install the latest updates from the develop branch:

.. code::

    pip install git+https://github.com/fredericjs/surfalize.git@develop
