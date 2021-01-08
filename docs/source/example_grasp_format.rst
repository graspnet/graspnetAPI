.. _example_grasp_format:

Grasp Label Format
==================

There are totally four kinds of data structures for grasp labels: Grasp, GraspGroup, RectGrasp and RectGraspGroup.
Their definitions can be found in grasp.py. The internal data format of each class is a numpy array.
Users can access or modify the value by provided functions.
Users can also manipulate the data directly but it is not recommended.
Please refer to the code for more details.


Loading a GraspGroup instance.

.. literalinclude:: ../../examples/exam_grasp_format.py
    :lines: 1-27

Users can access elements by index or slice.

.. literalinclude:: ../../examples/exam_grasp_format.py
    :lines: 29-35

Each element of GraspGroup is a Grasp instance.
The properties of Grasp can be accessed via provided methods.

.. literalinclude:: ../../examples/exam_grasp_format.py
    :lines: 37-46

RectGrasp is the class for rectangle grasps. The format is different from Grasp.
But the provided APIs are similar.

.. literalinclude:: ../../examples/exam_grasp_format.py
    :lines: 49-65