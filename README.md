apriltag
========

Extension of Apriltag library for use in BigDBee project https://bigdbee.hpcf.upr.edu/ to tag honeybees with tag25h5 tags.


Original version: http://april.eecs.umich.edu/media/apriltag/apriltag-2015-03-18.tgz from https://april.eecs.umich.edu/software/apriltag

Swatbotics changes: https://github.com/swatbotics/apriltag

- Added python wrappers and various improvements

BigDBee changes:

- Add new families: beetag, tag25h5, tag25h6
- Add inverse tag detection
- Adapt for Python3, fix memory leaks and segfault crashes
- Give access to QTP detection parameters in python code
- New batch script `apriltagdetect.py` to extract tags from video (in JSON format with extra information such as RGB and rectified tag image)


Dependencies
============

  - OpenCV

Building
========

    cd /path/to/apriltag
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j4
    
Running Demo
============

You can run `aprilag_opencv_demo` to do stuff, run with `-h` to get help.

So for example, you can run

    ./apriltag_opencv_demo -B    ../images/mapping_feb_2014/*.JPG
    ./apriltag_opencv_demo -B -c ../images/mapping_feb_2014/*.JPG

to benchmark the new code against the old code.

Running batch detection
=======================

Example for 1h videos at 20fps using inverted tag25h5 apriltags. Use 4 CPU threads. Save output in a single JSON file
```
python apriltagdetect.py -V video.mp4 -fps 20 -F tag25h5inv -f0 0 -f1 72100 -1 -D=0 -t 4 -m -rgb_mean
```
Notes: 

- fps is required to use OpenCV seek with CAP_PROP_POS_MSEC which seems more robust than accessing by frame number.

- OpenCV may use 100% CPU with multiple threads even with -t option. Some tests should be done before running batches in a shared CPU. Try -cvt option.

```
usage: apriltagdetect.py [-h] [-f FAMILIES] [-B N] [-t N] [-x SCALE]
                         [-b SIGMA] [-0] [-1] [-2] [-c] [-D DEBUG] [-I]
                         [-F FAMILY_PRESET] [-V VIDEO_IN] [-f0 F0] [-f1 F1]
                         [-fps FPS] [-outdir OUTDIR] [-tagout] [-tagplot] [-m]
                         [-mv MULTIFRAMEFILE_VERSION] [-cvt CV_NTHREADS]
                         [-max_side_length MAX_SIDE_LENGTH]
                         [-min_side_length MIN_SIDE_LENGTH]
                         [-min_aspect MIN_ASPECT] [-rgb_mean] [-tag_img]
                         [-tag_os TAG_OVERSAMPLING] [-tag_d TAG_D]

Detect apriltag in videos and images. Output to tagjson/ directory

optional arguments:
  -h, --help            show this help message and exit
  -f FAMILIES           Tag families (default tag36h11)
  -B N                  Tag border size in pixels (default 1)
  -t N                  Number of threads (default 4)
  -x SCALE              Quad decimation factor (default 1.0)
  -b SIGMA              Apply low-pass blur to input (default 0.0)
  -0                    Spend less time trying to align edges of tags
  -1                    Spend more time trying to decode tags
  -2                    Spend more time trying to precisely localize tags
  -c                    Use new contour-based quad detection
  -D DEBUG              define debug flags
  -I                    Process inverse image
  -F FAMILY_PRESET, --family-preset FAMILY_PRESET
                        Family preset: detect "inv" at the end of the name to
                        set both "family" and "inverse"
  -V VIDEO_IN           Input video
  -f0 F0                Frame start (default 0)
  -f1 F1                Frame end (default 0)
  -fps FPS              fps (default 20.0)
  -outdir OUTDIR        For video, basedir of output (default .)
  -tagout               Save image with detected tags overlay (default False)
  -tagplot              Plot tags overlay using matplotlib (default False)
  -m                    Save multiple frames into single JSON file (default
                        False)
  -mv MULTIFRAMEFILE_VERSION
                        Single JSON file version (default 0)
  -cvt CV_NTHREADS      Number of threads for OpenCV (default 2)
  -max_side_length MAX_SIDE_LENGTH
                        Maximum tag size in pixel (default 35)
  -min_side_length MIN_SIDE_LENGTH
                        Tags smaller than that are discarded (default 20)
  -min_aspect MIN_ASPECT
                        Tags smaller than that are discarded (default 0.7)
  -rgb_mean             Extract extra info rgb_mean (default False)
  -tag_img              Extract extra info tag image (default False)
  -tag_os TAG_OVERSAMPLING
                        Oversampling factor to extract tag image or compute
                        rgb_mean (default 1)
  -tag_d TAG_D          Size of tag (tag25h5 -> d=5) (default 5)
```

Acknowledgements
================

Original code: Edwin Olson, https://april.eecs.umich.edu/software/apriltag

Python wrapper and improvements: Matt Zucker, https://github.com/swatbotics/apriltag

Extra changes by Remi Megret are within the project "Large-scale multi-parameter analysis of honeybee behavior in their natural habitat" (https://bigdbee.hpcf.upr.edu/). This material is based upon work supported by the National Science Foundation under Grants No. 1707355 and 1633184.
