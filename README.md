# Bundle Adjustment for 3D Computer Vision

This project is used for bundle adjustment - a non-linear set of refinement of cameras for 3D computer vision. Given a set of measured images parameters, this method aims to create a reconstruction that minimizes the reprojection error.

This has been a very interesting, but less code-heavy project. However, the math required to understand the optimization algorithm is non-trivial. In order to solve the problem in a limited amount of time, the Levenberg-Marquardt algorithm must be implemented (for more info, check out this link: [Levenberg-Marquardt Algorithm] (http://people.duke.edu/~hpgavin/ce281/lm.pdf). It works well for nonlinear least squares. 

For further optimization to meet the required computational efficiency, one must also exploit the sparse structure of the matrices.

To use:
```
python3 eval_reconstruction.py --solution cube-solution.pickle
