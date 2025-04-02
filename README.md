Essentia
========

Essentia is an open-source C++ library for audio analysis and audio-based music information retrieval released under the Affero GPLv3 license. It contains an extensive collection of reusable algorithms which implement audio input/output functionality, standard digital signal processing blocks, statistical characterization of data, and a large set of spectral, temporal, tonal and high-level music descriptors. The library is also wrapped in Python and includes a number of predefined executable extractors for the available music descriptors, which facilitates its use for fast prototyping and allows setting up research experiments very rapidly. Furthermore, it includes a Vamp plugin to be used with Sonic Visualiser for visualization purposes. Essentia is designed with a focus on the robustness of the provided music descriptors and is optimized in terms of the computational cost of the algorithms. The provided functionality, specifically the music descriptors included in-the-box and signal processing algorithms, is easily expandable and allows for both research experiments and development of large-scale industrial applications.

Documentation online: http://essentia.upf.edu

Why this fork
-------------
This repository does not contain actual improvements in respect to the original algorithms developed in the main repository, but it focuses on 2 main aspects:
- compatibility with `numpy2`
- a better python packaging system

### How to create and install the essentia python package
#### Dependencies on MacOS
Install the main dependencies with [Brew](https://brew.sh/)
```zsh
% brew update
% brew install pkg-config gcc readline sqlite gdbm freetype libpng
% brew install eigen libsamplerate taglib libyaml fftw ffempeg@4 numpy
```
It is then important to link `ffmpeg@4` manually
```zsh
% brew unlink ffmpeg && brew link --force ffmpeg@4
```
#### Dependencies on Linux (test on Debian Bookworm and in a WSL)
Install the requirements:
```bash
$ sudo apt install curl cmake yasm pkg-config
```
All the others library on which `essentia` depends will be installed by a script
#### Packaging and Usage

To create a python source distribution run:
```zsh
% python3 setup.py sdist
```
this command will create a `tar.gz` archive in the `dist` folder that can directly be compiled and installed with `pip`.

After creating/activating a virtual environment (or a conda environment) run:

**MacOS**
```zsh
(venv)% ESSENTIA_WHEEL_SKIP_3RDPARTY=1 pip install dist/essentia-2.1b6.dev0.tar.gz
```
**Linux**
```bash
(venv)$ pip install dist/essentia-2.1b6.dev0.tar.gz
```
You are ready to use `essentia` in python


Quick start
-----------

Quick start using Python:
- http://essentia.upf.edu/documentation/essentia_python_tutorial.html
- [Jupyter Notebook Essentia tutorial](/src/examples/python/essentia_python_tutorial.ipynb)

Command-line tools to compute common music descriptors:
- [doc/sphinxdoc/extractors_out_of_box.rst](doc/sphinxdoc/extractors_out_of_box.rst)


Versions
--------

Official releases: https://github.com/MTG/essentia/releases

Github branches:
- [master](https://github.com/MTG/essentia/tree/master): latest updates; if you got any problem, try it first.

If you use example extractors (located in src/examples), or your own code employing Essentia algorithms to compute descriptors, you should be aware of possible incompatibilities when using different versions of Essentia.

