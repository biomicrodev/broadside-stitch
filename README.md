# Broadside

*Broadside* is an image processing pipeline used by the Laboratory for Bio-Micro Devices @ Brigham & Women's Hospital in Boston, MA. It consists of the following pre-processing steps:

1. pre-process
   1. make illumination profiles using [pybasic](https://github.com/biomicrodev/pybasic)
   2. make unmixing cache
   3. make corrected stacks
   4. register and stitch using [ASHLAR](https://github.com/labsyspharm/ashlar)
2. annotate, and view unmixing results
3. post-process
   1. nucleus and cell segmentation
   2. quantification and spillover correction
