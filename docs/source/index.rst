.. video2vision documentation master file, created by
   sphinx-quickstart on Sat Sep  3 15:35:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to video2vision's documentation!
========================================

.. image:: example.jpg
  :align: center

video2vision is a Python toolkit for processing multispectral images and video to estimate what an animal would see. It can be downloaded from:

`https://github.com/HanleyColorLab/video2vision <https://github.com/HanleyColorLab/video2vision>`_

video2vision provides a pre-built set of `operators <operators.html>`_ that implement a variety of image processing operations such as alignment, linearization, and conversion. These operators are assembled into a `pipeline <pipeline.html>`_, a `directed acyclic graph <https://en.wikipedia.org/wiki/Directed_Acyclic_Graph>`_ - or flowchart - encoding a full image processing procedure.

Quick Start Guide
=================

First, download and unzip the `libary <https://github.com/HanleyColorLab/video2vision/archive/refs/heads/main.zip>`_, and install it and its optional dependencies by running:

.. code-block:: bash

    python3 -m pip install .
    python3 -m pip install -r requirements-optional.txt

Second, start a Jupyter notebook server by running:

.. code-block:: bash

    jupyter notebook

This should open a notebook server in your web browser.

Navigate to the notebooks directory. You will need to first build an alignment pipeline using the `Alignment Pipeline Builder notebook <https://github.com/HanleyColorLab/video2vision/blob/main/notebooks/Alignment-Pipeline-Builder.ipynb>`_. Replace the paths in the first cell of the notebook and run the notebook. This will create a JSON file encoding the alignment pipeline.

Second, run the `Video Analysis notebook <https://github.com/HanleyColorLab/video2vision/blob/main/notebooks/Video-Analysis.ipynb>`_. The notebook will walk you through aligning, linearizing, and converting a video or set of still images. You can reuse the alignment pipeline created in the first notebook for further images and videos, rerunning the video analysis notebook for each one.

Defining a Pipeline
===================

A pipeline is defined in a JSON file. An `example pipeline <https://github.com/HanleyColorLab/video2vision/blob/main/data/video_alignment_pipeline.json>`_ is provided in the GitHub repo. JSON is a light-weight schema for encoding simple objects such as strings, numbers, dictionaries, and lists in a human-readable form. Conveniently, JSON syntax is very similar to Python syntax. For example, a list of numbers in JSON could be given as:

.. code-block:: json

    [1, 2, 3, 4]

And a dictionary could be given as:

.. code-block:: json

    {"a": 1, "b": 2}

However, JSON syntax is much more restrictive than Python syntax. For example, strings must be defined by double quotes (") in JSON; single quotes (') are not allowed. When in doubt, it may be best to define the pipeline you want to save in Python, then save using Python's built-in JSON parser:

.. code-block:: python

    pipeline = [...]

    import json

    with open('pipeline.json', 'w') as pipe_file:
        json.dump(pipeline, pipe_file)

A pipeline JSON consists of a list of dictionaries, where each dictionary encodes an operator. Each operator dictionary must have exactly three keys:

 * "index": The value should be a unique integer used to designate the operator.
 * "operator": This key defines the operator itself. The value should be a dictionary. The dictionary must have the key "class", whose value is the name of the class of :class:`Operator`, e.g. :class:`HorizontalFlip` or :class:`Translation`. (See below for a list of available operators). The dictionary may have other keys as well, whose contents are the arguments passed to the operator when it is initialized.
 * "edges": This key defines which operators the output of this operator is passed to. The value should be a (possibly empty) list of arbitrary length. Each entry in the list must be a list of length two. The first entry in the list is the index of the receiving operator. The second entry in the list is a dictionary with a single key, "in_slot". Most operators can receive only a single input, but a few, such as :class:`AutoTemporalAlign`, can receive multiple inputs. This dictionary is used to specify which input slot the output of this operator goes to. This is provided as a dictionary for future compatibility, in case we need to specify additional parameters for an edge.

Here is a very simple example of creating a pipeline that reads an image in, flips it horizontally, then writes it out again:

.. code-block:: python

    import json

    pipeline = [
        {
            "index": 0,
            "operator": {"class": "Loader"},
            "edges": [[1, {"in_slot": 0}]]
        }, {
            "index": 1,
            "operator": {"class": "HorizontalFlip"},
            "edges": [[2, {"in_slot": 0}]]
        }, {
            "index": 2,
            "operator": {"class": "Writer"},
            "edges": []
        }
    ]

    with open('pipeline.json', 'w') as pipe_file:
        json.dump(pipeline, pipe_file)

Once a :class:`Pipeline` has been created, it can be executed in Python:

.. code-block:: python

    import video2vision as v2v

    pipeline = v2v.load_pipeline('pipeline.json')
    pipeline.set_loader_paths('...')
    pipeline.set_writer_paths('...')
    pipeline.run()

Internals
=========

Internally, each :class:`Pipeline` is a :class:`networkx.DiGraph`, with each node representing an :class:`Operator`. Images are loaded from disk by one or more :class:`Loader` s, processed by operators, then written back to disk by one or more :class:`Writer` s.

Multiple images, or multiple frames in a video, are loaded into memory at a time. The number of images loaded is called the **batch size**, and can be set either as a parameter of the :class:`Loader`:

.. code-block:: python

    pipeline = [
        {
            "index": 0,
            "operator": {
                "class": "Loader",
                "batch_size": 32
            },
            "edges": [[1, {"in_slot": 0}]]
        }
    ]

Or set using the :meth:`video2vision.Pipeline.set_batch_size` method:

.. code-block:: python

    pipeline.set_batch_size(32)

The batch size should generally be as large as will fit comfortably into memory. The images are passed between operators as a dictionary. These dictionaries have only one mandatory key, 'image', whose value is a 4-dimensional :class:`numpy.ndarray` containing the batch of images. The array is indexed by (height, width, image or frame, band). The array has dtype float32, and is scaled to the range 0-1. Other keys may also be present, such as:

 * 'mask': Contains a 2-dimensional boolean mask indexed by (height, width). This provides a mask specifying which pixels in the images are valid.
 * 'names': Specifies the original names of the file(s) the images were taken from.

Contents
========

.. toctree::
   :maxdepth: 2

   operators
   io
   auto_operators
   pipeline
   utils
