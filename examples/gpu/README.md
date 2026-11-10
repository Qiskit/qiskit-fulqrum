# GPU example: ground-state energy on one or many nodes

`run_fulqrum_gpu_diagonalization.py` computes the lowest eigenvalues of a
molecular Hamiltonian in a tensor-product subspace, on GPUs with one MPI rank
per GPU. `run_fulqrum_gpu_diagonalization.sbatch` launches it with Slurm.

Both files are examples: adapt them to your problem and your cluster.

## Requirements

- Fulqrum built with the GPU module (see [`GPU_INSTALL.md`](../../GPU_INSTALL.md)).
- `mpi4py`, built against the MPI library that `srun` uses.
- One GPU per MPI rank.

## Inputs

- An FCIDUMP file for the molecule.
- A determinant file of alpha half strings, and optionally a separate one
  for beta (by default the alpha strings are used for both spins).

  A determinant file holds one bitstring per line, with orbital 0 the
  rightmost character. Every string in a file must have the same length (the
  number of spatial orbitals) and the same number of ones (the electrons of
  that spin).

## Running

```bash
srun --mpi=pmix -n 8 python3 run_fulqrum_gpu_diagonalization.py \
    --fcidump molecule.fcidump --alpha-dets alpha_dets.txt -r 1e-5
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--fcidump` | required | FCIDUMP file of the molecule |
| `--alpha-dets` | required | alpha half strings, one per line |
| `--beta-dets` | the alpha strings | beta half strings, one per line |
| `-r`, `--residual-norm` | `1e-3` | residual norm tolerance of the eigensolver |
| `--nroots` | `1` | number of eigenpairs |
| `--gpu-eigsh` | `0` | run the eigensolver's linear algebra on the GPU (`1`) or the host (`0`); matvecs run on the GPU either way |
| `--grid` | automatic | process grid `PxQ` (alpha × beta ranks), with `P*Q` equal to the number of ranks |
| `--rdm` | `0` | also compute the ground state's 1-RDM (`1`) or 1- and 2-RDMs (`2`) |
| `--out` | none | save the energies (and RDMs) to this `.npz` file |

The printed energies are total energies: the FCIDUMP's constant term is added
back. The eigensolver is PRIMME's `GD_Olsen_plusK` method, preconditioned
with the Hamiltonian's diagonal.

With `--out`, the `.npz` file holds `energies`, and with `--rdm` also
`gamma_a`, `gamma_b` (1-RDMs) and `G_aa`, `G_ab`, `G_bb` (2-RDMs), in PySCF's
`make_rdm12s(reorder=True)` convention.

## Slurm script

Edit the `#SBATCH` lines and the `FCIDUMP`, `ALPHA_DETS`, `BETA_DETS` and
`GRID` variables at the top of `run_fulqrum_gpu_diagonalization.sbatch`, then
`sbatch run_fulqrum_gpu_diagonalization.sbatch`.

The NCCL and UCX settings in the script performed well on an NVIDIA GB200 NVL72
cluster (4 GPUs per node, multi-node NVLink, InfiniBand adapters `ibp0`..`ibp3`).
On other systems start with only `FQ_DIST_BACKEND` and `CUDA_VISIBLE_DEVICES`,
and add network settings that suit your fabric. If host-side MPI over
InfiniBand does not work on your system, use `UCX_TLS=tcp` with
`UCX_NET_DEVICES=eth0`.
