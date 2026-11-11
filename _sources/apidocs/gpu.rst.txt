.. _fulqrum-gpu:

GPU
===

The GPU module provides matrix-free matrix-vector (matvec) product and reduced
density matrices (RDMs) compute capabilities for Chemistry Hamiltonians on Nvidia
GPUs. Coupled with the bundled distributed PRIMME eigensolver, it can diagonalize
molecular Hamiltonians on one node or multiple nodes with one MPI rank per GPU.
It requires a build with the GPU module enabled; see the GPU installation guide.

.. currentmodule:: fulqrum.gpu

Hamiltonians on tensor-product subspaces
----------------------------------------

For subspaces built as ``fq.Subspace([alpha_strs, beta_strs])``.

.. autosummary::
   :toctree: ../stubs/

   GPUHamiltonian
   DistributedMPIGPUHamiltonian
   TileGPUResidentHamiltonian

Hamiltonians on full-length determinant (full-strs) lists
---------------------------------------------------------

For subspaces given as a list of full determinants.

.. currentmodule:: fulqrum.gpu.full_strs

.. autosummary::
   :toctree: ../stubs/

   FullStrsGPUHamiltonian

Distributed eigensolver
-----------------------

.. currentmodule:: fulqrum.gpu

.. autosummary::
   :toctree: ../stubs/

   distributed_primme_eigsh

Configuration
-------------

GPU settings are read from ``FQ_*`` environment variables.

.. currentmodule:: fulqrum.gpu.config

.. autosummary::
   :toctree: ../stubs/

   GpuConfig

Logging and profiling
---------------------

.. currentmodule:: fulqrum.gpu

.. autosummary::
   :toctree: ../stubs/

   configure_gpu_logging
