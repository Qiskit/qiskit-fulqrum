
![](https://raw.githubusercontent.com/Qiskit/fulqrum/main/docs/images/fulqrum_logo.png)

# Fulqrum

![GitHub Release](https://img.shields.io/github/v/release/Qiskit/qiskit-fulqrum)
![PyPI Downloads](https://img.shields.io/pypi/dm/fulqrum)
![GitHub License](https://img.shields.io/github/license/Qiskit/qiskit-fulqrum)

A generalized framework for quantum subspace sparse linear algebra

Fulqrum is a set of tools for enabling the solution to large-scale Hamiltonian subspace sparse linear algebra problems over extended alphabets.  Fulqrum was designed specifically with the goals of: (1) Providing a unified code base for solving both spin and fermionic systems. (2) Working for arbitrary numbers of qubits. (3) Reducing memory consumption, as compared to existing methods. And finally, in the case of eigenvalue problems, decoupling operator construction from eigensolving itself.

Fulqrum also provides tools for generating compact subspaces that yield accurate solutions to quantum chemistry applications while potentially using orders of magnitude fewer bit-strings than standard quantum subspace methods alone.

Fulqrum can utilize both CPU and GPU (Nvidia only) computing resources. However, at present, not all features are supported across all computing modes.  See the table below for current feature availability:

| Feature | CPU-compatible | GPU-compatible |
| --- | --- | --- |
| Distributed (multi-node)| ✗ | ✓ |
| Max. qubits | 65,536 | 256 |
| Cartesian Subspace (TPB) | ✓ | ✓ |
| General Subspace (GDB) | ✓ | ✓ |
| 1-RDM and 2-RDM | ✗ | ✓ |
| Spin penality | ✗ | ✗ |
| Complex values | ✓ | ✗ |
| Generic qubit operators | ✓ | ✗ |
| Sample-based Quantum Diagonalization (SQD) compatible | ✓ | ✓ |


## Documentation

Docs: https://qiskit.github.io/qiskit-fulqrum/


## Installation


### Requirements

Outside of standard Python packages Fulqrum requires some Boost headers and OpenMP 3+.

Required Boost header files are included in the ``third-party/boost`` directory, and therefore, no explicit installation of Boost is needed.

### Git submodule

Fulqrum uses [`qiskit-addon-sqd-hpc`](https://github.com/Qiskit/qiskit-addon-sqd-hpc) and [`primme`](https://github.com/primme/primme) as Git submodules. Therefore, we can do either of the following to get actual and required files from those repos:

Clone Fulqrum with `--recurse-submodules` flag, and it will also clone files from the submodule.
```bash
git clone --recurse-submodules https://github.com/Qiskit/qiskit-fulqrum.git
```

OR

After cloning Fulqrum normally, do a submodule update.
```bash
git clone https://github.com/Qiskit/qiskit-fulqrum.git
cd fulqrum
git submodule update --init --recursive
```

### Building files locally

In order to run the unittests locally, it is only necessary to build the Cython files inplace:

```bash
python setup.py build_ext --inplace
```
you can add also use env flags such as `CC=clang CXX=clang++` or set the target architecture using `FQ_ARCH=znver4` (or whatever you have)

### Installation on Linux

Installation on Linux is simple:

```bash
pip install .
```

### Installation on OSX

On OSX, to get OpenMP, one should install llvm using homebrew:

```bash
brew install llvm
```
and the following, or something similar, added to the users `.zshrc` file:

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
```

Then installation of Fulqrum with openmp can be accomplished using a call like:

```bash
CC=clang CXX=clang++ pip install .
```

### Parallel build

You can turn on parallel compilation by setting the environment variable
`FQ_BUILD_PARALLEL=<num threads>`, which can accelerate the build time. Example
with `8` parallel threads:

```bash
# OSX
FQ_BUILD_PARALLEL=8 CC=clang CXX=clang++ pip install .

# Linux
FQ_BUILD_PARALLEL=8 pip install .

# local
FQ_BUILD_PARALLEL=8 python setup.py build_ext --inplace
```

### Runtime performance tuning

The type-2 build/matvec scripts (`csrlike_builder2`, `csr2`,
`matvec2`) allows two optional **runtime** environment variables for tuning. Both have defaults and are read at run time (no rebuild needed), so they are safe to leave unset.

| Variable | Default | Description |
| --- | --- | --- |
| `FQ_BLK` | `128` | The number of rows in a block of rows. FQ_BLK number of rows reuses a chunk of group_ladder_ptrs. Larger blocks give more reuse but risk overflowing the per-core cache; smaller blocks reduce reuse. `128` is a good middle ground. Values `<= 0` are ignored. |
| `FQ_LADDER_WIDTH` | `2` | Number of ladder bits used to bucket each group's terms, giving `2^FQ_LADDER_WIDTH` bins per group (so `2` → 4 bins, `4` → 16 bins). |

Both variables tune the speed only. They change no result.

Example: run a workload with a larger row block and finer ladder bucketing:

```bash
FQ_BLK=256 FQ_LADDER_WIDTH=4 python your_workload.py
```


## Notes on `Subspace` input format

Two modes of input to subspace is supported:
1. **Half-string mode:** In this mode, the input `subspace_strs` to the `Subspace()` is a length-2
`tuple[list[str], list[str]]`, where the first element represents _alpha_ strings, and the second
one represents _beta_ strings. Internally, `fulqrum` will sort each list and perform a Cartesian product between the two lists to construct the full subspace. A full bitstring will be represented by `beta half str + alpha half str`, and the final length of subspace bitstring will be `len(beta half string) + len(alpha half string)`. For example, `[['100', '001, '010'], ['110', '000']]` is a valid input `subspace_strs` in half-string mode. The final subspace size will be `2 * 3 = 6`, with each bitstrings with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications. For instance, many popular chemistry packages such as PySCF takes alpha and beta strings separately and extends the subspace by taking Cartesian product. This mode enables PySCF-like functionality and keeps the memory requirement in check.

2. **Full-string mode:** In this mode, `subspace_strs` has to be length-1 `tuple[list[str]]`. In this mode, no Cartesian product is taken internally, and the supplied _full_ bitstrings are used as is after sorting.

**Note:** There is no special flag to denote half-string vs. full-string mode. `Fulqrum` will compute the length of the input `subspace_strs` and detect half-string or full-string mode automatically to construct the subspace. If `subspace_strs` has a different length other than 1 or 2 or its element(s) is not `list[str]`, it will throw an `TypeError`.


## Benchmarks

Benchmarks for select quantum chemistry problems comparing the performance of Fulqrum to [`Dice`](https://github.com/sanshar/Dice) and [`SBD`](https://github.com/r-ccs-cms/sbd).

Benchmarks target finding the ground-state energy within the given subspace dimension.  Solvers configured so that they return the same energy to a relative tolerance of 1E-7.  Here both Fulqrum and SBD are used in matrix-free mode.

### CPU results

#### Fe4S4 (36 orbitals, 54 electrons) (arXiv:2405.05068)

12-core, 24-thread AMD 7900, 128 Gb memory

| Solver | Subspace dimension | Runtime (s) | Memory (Gb) | Speedup |
| --- | --- | --- | --- | --- |
| Dice | 1E6 | 93.6 | 47.4 | |
| Fulqrum - primme | 1E6 | 7.4 | 3.8 | 11x |
| Fulqrum - scipy | 1E6 | 8.1 | 3.8 | 10x |
| SBD | 1E6 | 82.2 | 1.2 |  |
| --- | --- | --- | --- | --- |
| Dice | 1E8 | - | OOM | |
| Fulqrum - primme | 1E8 | 405 | 13.4 | 99x |
| Fulqrum - scipy | 1E8 | 527 | 24.6 | 76x |
| SBD | 1E8 | 40225 | 82.2 |  | 


Dual-socket Intel Xeon Platinum 8474C (48-core, 96 threads), 2 TB memory

| Solver | Subspace dimension | Runtime (s) | Memory (Gb) | Speedup |
| --- | --- | --- | --- | --- |
| Dice | 1.6E7 | 589 | 436 | |
| Fulqrum - primme | 1.6E7 | 45 | 5 | 13x |
| SBD | 1.6E7 | 754 | 3 | |
| --- | --- | --- | --- | --- |
| Dice | 1E8 | - | OOM | |
| Fulqrum - primme | 1E8 | 149 | 12 | 38x |
| SBD | 1E8 | 5705 | 20 | |


### GPU results

Results run on GB200 cluster; 4-GPUs per node

#### Molten salt molecules (arXiv:2606.30402)

| Molecule | (norb, nelec) | Subspace dimension | Fulqrum time (s) | SBD time (s) | Speedup |
| --- | --- | --- | --- | --- | --- |
| conf3-clust0021 | (30, 12) | 2E9 | 40.7 | 237.8 | 5.8x |
| conf5-clust0015 | (29, 10) | 5.8E9 | 110.3 | 1305.4 | 11.9x |
| conf8-clust0017 | (29, 10) | 5.8E9 | 127 | 2393 | 18.8x |
| FLiBe | (29, 10) | 5.4E8 | 60.2 | 1200 | 19.9x |

#### Lysozyme fragments (arXiv:2605.01138)

| Molecule | (norb, nelec) | Subspace dimension | Fulqrum time (s) | SBD time (s) | Speedup |
| --- | --- | --- | --- | --- | --- |
| atom_10 | (38, 26) | 8.9E9 | 93.9 | 422.4 | 4.5x |
| atom_17 | (38, 24) | 4.7E9 | 68.4 | 172.4 | 2.5x |
| atom_19 | (39, 24) | 5.5E9 | 74.4 | 218.3 | 2.9x |
| atom_83 | (34, 36) | 7.6E9 | 80 | 461.5 | 5.8x |
| atom_333 | (34, 36) | 7.6E9 | 78.7 | 499.1 | 6.3x |
| atom_444 | (34, 36) | 7.9E9 | 81.6 | 439.4 | 5.4x |
| atom_1590 | (34, 34) | 5.1E9 | 63 | 243.6 | 3.9x |
| atom_1795 | (35, 24) | 6.5E9 | 74.3 | 287.5 | 3.9x |
| atom_1799 | (35, 26) | 7.2E9 | 77.5 | 326.4 | 4.2x |
| atom_1859 | (37, 24) | 4.5E9 | 63.6 | 148.9 | 2.3x |
| atom_2014 | (34, 36) | 1.6E10 | 147.8 | 1207.4 | 8.2x |
| atom_2023 | (34, 36) | 1.4E10 | 130.9 | 947.8 | 7.2x |
| atom_2200 | (35, 38) | 1.6E10 | 150.3 | 1237.2 | 8.2x |
| atom_2209 | (35, 38) | 1.8E10 | 157.5 | 1316 | 8.4x |
| atom_2219 | (34, 36) | 7.7E9 | 80.7 | 489.5 | 6.1x |
| atom_2436 | (34, 36) | 6.6E9 | 75 | 359.2 | 4.8x |
| atom_2457 | (37, 26) | 9.4E9 | 96.6 | 512.3 | 5.3x |
| atom_2539 | (34, 36) | 1.5E10 | 134.5 | 1038.5 | 7.7x |
| atom_2548 | (35, 38) | 1.7E10 | 150.1 | 1314.3 | 8.8x |
| atom_2580 | (35, 38) | 9.2E9 | 89.8 | 621.5 | 6.9x |
| atom_1752 | (45, 46) | 3.5E9 | 90 | 582.3 | 6.5x |

* SBD numbers used commit [95bf45a](https://github.com/r-ccs-cms/sbd/commit/95bf45a5be9ffd977fac7b0aaa359833caf3401a) or earlier.