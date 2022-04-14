# fix-perspective

> Tool to compensate perspective distortion in document images

* Does align horizontal + vertical
* Needs `opencv4` and `eigen3`

## Installation

Install system dependencies (`opencv4` and `eigen3`). On Ubuntu (and derivates), you can run

    sudo make deps-ubuntu

To verify all system dependencies are met, run

    make check

To just build the program, run

    make

To also install into $PREFIX, run

    make install

## Usage

Reads the input image file under the first argument, writes its output image file under the second argument. (File name suffix determines image format.)

Example:

    fix-perspective input_image.tif output_image.png

## Limits

- **not thorougly tested yet**
- for best results use cropped scans
- text is typeset mostly in regular lines, esp. if there are multiple columns
- *Warning:* output files are *not* suitable for archival. Metadata could be lost.
