.. error-parity documentation master file, created by
   sphinx-quickstart on Thu Nov 23 16:53:42 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to error-parity's documentation!
========================================

The :code:`error-parity` package allows you to easily achieve error-rate 
fairness between societal groups.
It's compatible with any score-based predictor, and can map out all of its 
attainable fairness-accuracy trade-offs.

Full code available on the `GitHub repository`_, 
including various `jupyter notebook examples`_ .

Check out the following sub-pages:

.. toctree::
   :maxdepth: 1

   Readme file <readme>
   API reference <modules>
   Example notebooks <notebooks>


Citing
------

The :code:`error-parity` package is the basis for the following `publication`_:

.. code-block:: bib

   @misc{cruz2023unprocessing,
         title={Unprocessing Seven Years of Algorithmic Fairness}, 
         author={Andr{\'{e}} F. Cruz and Moritz Hardt},
         year={2023},
         eprint={2306.07261},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }

All additional supplementary materials are available on the `supp-materials`_ branch of the `GitHub repository`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _GitHub repository: https://github.com/socialfoundations/error-parity
.. _jupyter notebook examples: https://github.com/socialfoundations/error-parity/tree/main/examples
.. _publication: https://arxiv.org/abs/2306.07261
.. _supp-materials: https://github.com/socialfoundations/error-parity/tree/supp-materials
