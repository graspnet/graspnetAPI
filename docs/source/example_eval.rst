.. _example_eval:

Evaluation
==========

Data Preparation
^^^^^^^^^^^^^^^^

The first step of evaluation is to prepare your own results.
You need to run your code and generate a `GraspGroup` for each image in each scene.
Then call the `save_npy` function of `GraspGroup` to dump the results.

To generate a `GraspGroup` and save it, you can directly input a 2D numpy array for the `GraspGroup` class:
::

  gg=GraspGroup(np.array([[score_1, width_1, height_1, depth_1, rotation_matrix_1(9), translation_1(3), object_id_1],
                          [score_2, width_2, height_2, depth_2, rotation_matrix_2(9), translation_2(3), object_id_2],
                          ...,
                          [score_N, width_N, height_N, depth_N, rotation_matrix_N(9), translation_N(3), object_id_N]]
                ))
  gg.save_npy(save_path)

where your algorithm predicts N grasp poses for an image. For the `object_id`, you can simply input `0`. For the meaning of other entries, you should refer to the doc for Grasp Label Format-API Loaded Labels

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

You can choose to generate dump files for only one camera, there will be no error for doing that.

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


