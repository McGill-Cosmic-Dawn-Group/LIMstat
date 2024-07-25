***************************************
``limstat``: LIM observation simulator
***************************************

The ``limstat`` package provides all of the tools and data structures
required to simulate interferometric line-intensity mapping observations,
including cosmological signal, noise, foregrounds, and telescope effects.
It can 2D and 3D maps (coeval cubes), as well as compute spherical and
cylindrical power spectra and associated window functions.

For usage examples and documentation, see ReadTheDocsTBD.

.. inclusion-marker-installation-do-not-remove

Installation
============

For users
---------

The package is installable, along with its dependencies, with PyPi. We
recommend using Anaconda and creating a new conda environment before
installing the package from the GitHub repository:

::

   $ conda create -n limstat python=3
   $ conda activate limstat
   $ git clone https://github.com/McGill-Cosmic-Dawn-Group/LIMstat.git
   $ cd LIMstat
   $ python3 -m pip install .

This will install required dependencies. 
New versions are frequently released on PyPi.

For developers
--------------

We recommend installing the package with the developer option of PyPi, that is
::

   $ python3 -m pip install -e .[dev]

Dependencies
^^^^^^^^^^^^

If you are using ``conda``, you may wish to install the following
dependencies manually to avoid them being installed automatically by
``pip``:

::
   $ conda install -c conda-forge numpy astropy cached_property scipy healpy uvtools matplotlib deprecated


Running Tests
^^^^^^^^^^^^^

Uses the ``pytest`` package to execute test suite. From the source
``limstat`` directory run: ``pytest``.

.. exclusion-marker-installation-do-not-remove

Running ``limstat``
======================

See the documentation and the tutorials (dedicated folder) for an overview and
examples of how to run ``limstat``.

