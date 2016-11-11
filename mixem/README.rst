This is EE511 Project 4
All of the Project based on Mixem Package

Problem 1 (Mix Gaussian + EM + Comparison)
==========================================
mixem/examples/MixGaussian.py

Problem 2 (Fitting Faithful Data)
==========================================
mixem/examples/FixData.py
(Generating Fitting_Data_Contour.html)

Extra (Noise EM)
==========================================
mixem/examples/TestNoise.py

What's Used else
==========================================
mixem/mixem/em.py
mixem/mixem/Nem.py
mixem/mixem/progress.py
mixem/mixem/model.py
mixem/mixem/distribution/normal.py


Below remains same
mix'EM 
======


mixem is a pure-python implementation of the Expectation-Maximization (EM) algorithm for fitting mixtures of probability distributions. It works in Python 2 and Python 3 (tested with 2.7 and 3.5.1) and uses few dependencies (only NumPy and SciPy).


.. image:: http://i.imgur.com/kJgsHMG.png
   :scale: 50 %
   :alt: Old Faithful example
   :align: left


Features
--------

* Easy-to-use and fully-documented API
* Built-in support for several probability distributions
* Easily define custom probability distributions by implementing their probability density function and weighted log-likelihood

Documentation
-------------
Find the mix'EM documentation on `ReadTheDocs <https://mixem.readthedocs.org/en/latest/>`_.


Installation
------------

::

    pip install mixem
