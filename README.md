## video2vision

<img align="center" src="docs/source/example.jpg">

video2vision is an image processing toolkit for using multispectral videos to approximate the vision of animals. The toolkit provides support for aligning videos from separate cameras, linearizing to remove camera post-processing, and converting to animal vision.

To use video2vision, install the latest version by running:

```
pip install video2vision
```

# Quick Start Guide

First, download and unzip the libary by clicking Code at the top of this page, then Download Zip. Unzip the library and install it and its optional dependencies by running:

```
python3 -m pip install .
python3 -m pip install -r requirements-optional.txt
```

Second, start a Jupyter notebook server by running:

```
jupyter notebook
```

This should open a notebook server in your web browser.

Navigate to the notebooks directory. You will need to first build an alignment pipeline using the [Alignment Pipeline Builder notebook](https://github.com/HanleyColorLab/video2vision/blob/main/notebooks/Alignment-Pipeline-Builder.ipynb). Replace the paths in the first cell of the notebook and run the notebook. This will create a JSON file encoding the alignment pipeline.

Second, run the [Video Analysis notebook](https://github.com/HanleyColorLab/video2vision/blob/main/notebooks/Video-Analysis.ipynb). The notebook will walk you through aligning, linearizing, and converting a video or set of still images. You can reuse the alignment pipeline created in the first notebook for further images and videos, rerunning the video analysis notebook for each one.
