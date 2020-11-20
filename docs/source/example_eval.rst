.. _example_eval:

Evaluation
==========

Data Preparation
^^^^^^^^^^^^^^^^

The first step of evaluation is to prepare your own data.
You need to run your code and generate a `GraspGroup` for each image in each scene.
Then call the `save_npy` function of `GraspGroup` to dump the results.

The file structure of dump folder should be as follows:

::

    |-- dump_folder
        |-- scene_0100
        |   |-- kinect                  
        |   |   |
        |   |   --- 0000.npy to 0255.npy
        |   |    
        |   --- realsense
        |       |
        |       --- 0000.npy to 0255.npy
        |
        |-- scene_0101
        |
        ...
        |
        --- scene_0189

You can only generate dump for one camera, there will be no error for doing that.

Evaluation API
^^^^^^^^^^^^^^

Get GraspNetEval instances.

.. literalinclude:: ../../examples/exam_eval.py
    :lines: 4-17

Evaluate A Single Scene
-----------------------

.. literalinclude:: ../../examples/exam_eval.py
    :lines: 19-23

Evaluate All Scenes
-------------------

.. literalinclude:: ../../examples/exam_eval.py
    :lines: 25-27

Evaluate 'Seen' Split
---------------------

.. literalinclude:: ../../examples/exam_eval.py
    :lines: 29-31


