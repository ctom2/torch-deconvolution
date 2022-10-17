# Image deconvolution algorithms
Implementation of image deconvolution algorithms using `torch`.

Values of an input image need to be in range [0,1] and the summation of PSF elements has to equal to 1.

## Overview

☑️ Richardson-Lucy iteration [1,2]</br>
☑️ Richardson-Lucy iteration with Total Variation (TV) [7]</br>
☑️ Landweber iteration [3]</br>
❌ Wiener filter [4]</br>
❌ Split Bergman method [5]</br>
❌ ADMM [6]</br>

## References

[1] William Hadley Richardson, "Bayesian-Based Iterative Method of Image Restoration*," J. Opt. Soc. Am. 62, 55-59 (1972).

[2] Lucy, L. B., “An iterative technique for the rectification of observed distributions”, *The Astronomical Journal*, vol. 79, p. 745 (1974). doi:10.1086/111605.

[3] Landweber, L., "An iteration formula for Fredholm integral equations of the first kind", Amer. J. Math. 73, 615–624 (1951).

[4] Wiener, Norbert, et al., "Extrapolation, interpolation, and smoothing of stationary time series: with engineering applications", Vol. 113. No. 21. Cambridge, MA: MIT press, (1949).

[5] Goldstein, Tom, and Stanley Osher. "The split Bregman method for L1-regularized problems." SIAM journal on imaging sciences 2.2 (2009): 323-343.

[6] Afonso, Manya V., José M. Bioucas-Dias, and Mário AT Figueiredo. "An augmented Lagrangian approach to the constrained optimization formulation of imaging inverse problems." IEEE transactions on image processing 20.3 (2010): 681-695.

[7] Dey, Nicolas, et al. "Richardson–Lucy algorithm with total variation regularization for 3D confocal microscope deconvolution." Microscopy research and technique 69.4 (2006): 260-266.
