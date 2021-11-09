# StreaMRAK
Streaming, multi-resolution, adaptive, kernel solver

Cite the code: [![DOI](https://zenodo.org/badge/383231702.svg)](https://zenodo.org/badge/latestdoi/383231702)

## Overview
The StreaMRAK folder contains the StreaMRAK code, while the MRFALKON folder contains the LP-KRR code. 
Experiments for the StreaMRAK paper are found in the Experiments folder. To run the experiments, it is first neccessary 
to generate the datasets using the code in the Experiments/Datasets folder. The MRFALKON code corresponds to the LP-KRR in the StreaMRAK paper.

When running the MRFALKON experiments, it is neccessary to set the correct value for the
lowlim_ntrp parameter in the MRFALKONmainConfig_mrfalkon configuration file. This file can be 
found in the folder StreaMRAK/StreaMRAKconfig.

For the varsin experiments
lowlim_ntrp = 110000

For the dumbell experiments
lowlim_ntrp = 180000

For the double pend experiments
lowlim_ntrp = 390000

## Note
The falkonSolver.py implementation in the StreaMRAK folder is a python adaptation that we have made of the MATLAB code in 
https://github.com/LCSL/FALKON_paper.git 
with the affiliated paper [[1]](#1).


## References
<a id="1">[1]</a> 
A. Rudi, L. Carratino, L. Rosasco, FALKON: An Optimal Large Scale Kernel Method, in: Advances in Neural 
Information Processing Systems Vol. 30, 2017, pp. 3889-3899.
