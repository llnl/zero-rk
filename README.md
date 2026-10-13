# Zero-RK : Zero Order Reaction Kinetics

Zero-RK is a software package that simulates chemically reacting systems using sparse, preconditioned, adaptive matrix methods to achieve orders-of-magnitude reduction in simulation time while maintaining accurate results.

Authors
----------------
Zero-RK was created by [Matthew McNenly](https://people.llnl.gov/mcnenly1), with significant contributions from [Russell Whitesides](https://people.llnl.gov/whitesides1) and [Simon Lapointe](https://people.llnl.gov/lapointe2).

Installation Notes
----------------

Zero-RK builds with cmake and depends on cmake version 3.12 or higher.  Zero-RK also requires a C11/C++14 compatible compiler suite, and has been tested with gnu, intel, and clang compilers on Linux and clang on Mac OS.  The basic installation process is:

    $ git clone https://github.com/llnl/zero-rk   #git
    $ cd zero-rk
    $ mkdir build
    $ cd build
    $ cmake ../                                   #configure
    $ make                                        #build
    $ ctest                                       #test
    $ make install                                #install

This "vanilla" build process will download and install most dependencies (including `sundials`, `superlu`, `superlu_dist`, and `spify`).  The user will need to supply a working MPI implementation as well as blas and lapack libraries.  The core code base can be built without MPI by invoking cmake as `cmake ../ -DENABLE_MPI=OFF`, however this will disable parallel applications including the flame solvers and global sensitivity analysis (GSA) codes.  The build will attempt to use the Intel Math Kernel Libraries (MKL) if the environment variable MKLROOT is set.

### Parallel Builds

To speed up the build process, parallel builds can be enabled for the build dependencies by setting the environment variable `CMAKE_BUILD_PARALLEL_LEVEL` to the number of available cores/threads before invoking `cmake`.  Parallel builds of the `zero-rk` source can be enabled with `make -j $np` at the build step.

### Optional System Installed Libraries

*Optionally*, for each of `SUNDIALS`, `SUPERLU`, and `SUPERLU_DIST` the user may supply their own system installed version of the relevant packages by specifying, for example, `-DSYSTEM_SUNDIALS_ROOT=/path/to/sundials/install` as an option to the `cmake` command.  Supported versions of packages are shown below:

| package            |           versions          |
| -------            |           --------          |
| `sundials`         | 2.7.0, 3.2.1, 4.1.0, 5.8.0  |
| `SuperLU`          | 5.3.0                       | 
| `SuperLU_DIST`     | 6.4.0                       | 

Other versions may work, but have not been tested.  For older `sundials` versions the cmake variable `SUNDIALS_VERSION` needs to be set to the appropriate major version number (e.g. `cmake ../ -DSUNDIALS_VERSION=4`).

### Tests

The test suite can be run with `ctest` after the build completes.  If everything is working correctly all tests should pass.

### Install

The applications, libraries, and examples should be installed to `CMAKE_INSTALL_PREFIX` with `make install`.  The examples are only properly configured when installed.  The default path for `CMAKE_INSTALL_PREFIX` is `build/inst_dir` but this can be changed by providing a different path (`cmake ../ -DCMAKE_INSTALL_PREFIX=/path/to/install/zerork`).

Applications
----------------

The current release includes applications in five major categories:
 - Zero-dimensional, constant- or variable-volume reactors with variants for batched and parallel computation as well as global sensitivity analysis.  Also included are an accelerated sensitiviy analysis tool using the tangent linear approximation (based on work of Almohammadi et al: https://doi.org/10.1016/j.combustflame.2021.111426) and a solver for perfectly stirred reactors:
   - `constVolumeWSR`
   - `constVolumeWSR_TLA`
   - `constVolumePSR`
   - `perturbAFactor`
   - `perturbAFactorGSA`
   - `variable_volume`
   - `variable_volume_batch`
   - `variable_volume_gsa`
- One-dimensional, laminar, premixed-, diffusion-, and counterflow-flames solvers:
   - `premixed_steady_flame_solver`
   - `premixed_unsteady_flame_solver`
   - `diffusion_steady_flame_solver`
   - `diffusion_unsteady_flame_solver`
   - `counterflow_steady_flame_solver`
   - `counterflow_unsteady_flame_solver`
 - Mechanism analysis/debugging tools:  
   - `thermo_check`
   - `idt_diagnostic`
 - A plugin for coupling to reacting CFD: `cfd_plugin`
 - A tool to optimize kinetic rate parameters to tune a reduced model to match a detailed model: `rate_optimization`

Examples for all the applications are installed at `${CMAKE_INSTALL_PREFIX}/share/zerork/examples`.  The inputs for most applications are in YAML format and options are described in the example input files.  The examples also include reference output so that the user can verify that their installation is working as expected.


Citing
----------------

If you use Zero-RK in a scholarly article, please cite the article(s) describing the application(s) used. 

- For zero-dimensional solvers:

> M.J. McNenly, R.A. Whitesides, and D.L. Flowers, Faster solvers for large kinetic mechanisms using adaptive preconditioners. Proceedings of the Combustion Institute, 35(1) (2015) 581-587. https://doi.org/10.1016/j.proci.2014.05.113

- For variable-volume solvers:

> S. Cheng, D. Kang, A. Fridlyand, S.S. Goldsborough, C. Saggese, S. Wagnon, M.J. McNenly, M. Mehl, W.J. Pitz, D. Vuilleumier, Autoignition behavior of gasoline/ethanol blends at engine-relevant conditions, Combustion and Flame. 216 (2020) 369-384. https://doi.org/10.1016/j.combustflame.2020.02.032.

- For the global sensitivity analysis solvers:

> A. Fridlyand, M.S. Johnson, S.S. Goldsborough, R.H. West, M.J. McNenly, M. Mehl, W.J. Pitz, The role of correlations in uncertainty quantification of transportation relevant fuel models, Combustion and Flame. 180 (2017) 239-249. https://doi.org/10.1016/j.combustflame.2016.10.014.

- For one-dimensional, laminar flames solvers:

> S. Lapointe, R.A. Whitesides, and M.J. McNenly, Sparse, iterative simulation methods for one-dimensional laminar flames. Combustion and Flame, 204 (2019) 23-32. https://doi.org/10.1016/j.combustflame.2019.02.030.   
> S. Lapointe, Y. Xuan, H. Kwon, R.A. Whitesides, M.J. McNenly, A computationally-efficient method for flamelet calculations, Combustion and Flame. 221 (2020) 94-102. https://doi.org/10.1016/j.combustflame.2020.07.035.

- For the mechanism analysis/debugging tools:

> N.J. Killingsworth, M.J. McNenly, R.A. Whitesides, and S.W. Wagnon, Cloud based tool for analysis of chemical kinetic mechanisms, Combustion and Flame. 221 (2020) 170-179. https://doi.org/10.1016/j.combustflame.2020.06.010.

> 

License
----------------

Zero-RK is distributed under the terms of the BSD-3-Clause license.

All new contributions must be made under the same license.

See LICENSE and NOTICE for details.

SPDX-License-Identifier: (BSD-3-Clause)

LLNL-CODE-2025129
