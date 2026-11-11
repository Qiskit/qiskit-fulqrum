#!/usr/bin/env bash
# Example: Fulqrum ground-state solve on NODES x 4 GPUs, one MPI rank per GPU.
#
# Edit the #SBATCH lines and the paths below for your cluster, then
#   sbatch run_fulqrum_gpu_diagonalization.sbatch
#
# The network settings target NVIDIA GB200 NVL72 systems: 4 GPUs per node,
# multi-node NVLink, and InfiniBand adapters named ibp0..ibp3. On other
# systems start with only FQ_DIST_BACKEND and CUDA_VISIBLE_DEVICES, and
# adapt the rest to your network.

#SBATCH --job-name=fulqrum-gpu-diag
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Optional: keep all nodes in same MNNVL domain (e.g., on NVL72 cluster), on
# Slurm topology plugins that support it. Set the value equal to --nodes.
# Your cluster may be different.
##SBATCH --segment=2

set -euo pipefail

FCIDUMP=/path/to/fcidump.txt
ALPHA_DETS=/path/to/alpha_dets.txt
# Optional; leave empty to use the alpha strings for beta too.
BETA_DETS=
# Process grid PxQ with P*Q = nodes * 4; leave empty for an automatic choice.
GRID=

EXTRA=()
[ -n "$BETA_DETS" ] && EXTRA+=(--beta-dets "$BETA_DETS")
[ -n "$GRID" ] && EXTRA+=(--grid "$GRID")

export SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}

srun --mpi=pmix --kill-on-bad-exit=1 --cpu-bind=cores bash -c '
  # One GPU per rank.
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

  # EXAMPLE only.
  # Following combination of NCCL flags
  # gave better performance on GB200 NVL72 cluster.
  # Tweak for your cluster.
  export FQ_DIST_BACKEND=nccl
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_IB_HCA=ibp
  export NCCL_NET_PLUGIN=none
  export NCCL_NET_GDR_C2C=1
  export NCCL_CUMEM_ENABLE=1
  export NCCL_MNNVL_ENABLE=1
  export NCCL_NVLS_ENABLE=1
  export NCCL_SHM_DISABLE=0
  export NVIDIA_IMEX_CHANNELS=0
  export NCCL_CROSS_NIC=1
  export NCCL_IB_ADAPTIVE_ROUTING=1
  export NCCL_NCHANNELS_PER_NET_PEER=16
  export NCCL_IB_QPS_PER_CONNECTION=16
  export NCCL_IB_SPLIT_DATA_ON_QPS=1
  export NCCL_P2P_NET_CHUNKSIZE=2097152
  export NCCL_BUFFSIZE=16777216

  # Host-side MPI (e.g., eigensolver reductions) over InfiniBand.
  # If your fabric differs, the portable fallback is
  # UCX_TLS=tcp with UCX_NET_DEVICES=eth0.
  export UCX_TLS=rc_mlx5,ud_mlx5,dc_mlx5,sm,self
  export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1
  export OMPI_MCA_coll_hcoll_enable=0
  export PMIX_MCA_gds=^ds12
  export PMIX_MCA_psec=native

  # CPU threads per rank for operator construction and BLAS.
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores

  exec python3 "$SCRIPT_DIR/run_fulqrum_gpu_diagonalization.py" "$@"
' bash --fcidump "$FCIDUMP" --alpha-dets "$ALPHA_DETS" --residual-norm 1e-3 \
       --gpu-eigsh 0 "${EXTRA[@]}"
