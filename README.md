# Bundle Adjustment
Bundle Adjustment for 3D Vision

This project is used for bundle adjustment - a non-linear set of refinement of cameras for 3D computer vision. Given a set of measured images parameters, this method aims to create a reconstruction that minimizes the reprojection error.

To use:
```
python3 eval_reconstruction.py --solution cube-solution.pickle
