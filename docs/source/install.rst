Installation
============

.. note::
    
    Only Python 3 on Linux is supported.

Prerequisites
^^^^^^^^^^^^^

Python version under 3.6 is not tested. 

Dataset
^^^^^^^

Download
--------

Download the dataset at https://graspnet.net/datasets.html

Unzip
-----

Unzip the files as shown in https://graspnet.net/datasets.html.

Rectangle Grasp Labels
----------------------
Rectangle grasp labels are optional if you need labels in this format.
You can both generate the labels or download the file_. 

If you want to generate the labels by yourself, you may refer to :ref:`example_generate_rectangle_labels`.

.. note::
    
    Generating rectangle grasp labels may take a long time.

After generating the labels or unzipping the labels, you need to run copy_rect_labels.py_ to copy rectangle grasp labels to corresponding folders.

.. _copy_rect_labels.py: https://github.com/graspnet/graspnetAPI/blob/master/copy_rect_labels.py

.. _file: https://graspnet.net/datasets.html

Dexnet Model Cache
------------------

Dexnet model cache is optional without which the evaluation will be much slower(about 10x time slower).
You can both download the file or generate it by yourself by running gen_pickle_dexmodel.py_(recommended). 

.. _gen_pickle_dexmodel.py: https://github.com/graspnet/graspnetAPI/blob/master/gen_pickle_dexmodel.py

Install API
^^^^^^^^^^^
You may install using pip::
    
    pip install graspnetAPI

You can also install from source::

    git clone https://github.com/graspnet/graspnetAPI.git
    cd graspnetAPI/
    pip install .
