# fix-perspective

> Tool to correct perspective distortion

* Does align horizontal + vertical
* Uses opencv4 and eigen3

## Installation

Run `make`

## Usage

Example: `./fix-perspective input_image output_image`

## Limits

- not thorougly tested yet
- for best results use cropped scans
- text "in register" if multiple columns
- *Warning:* output files are *not* suitable for archival. Metadata could be lost.
