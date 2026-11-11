# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import logging
import time

import cupy
import numpy as np

from ..common.wide import check_width, popcount_w
from ..distributed.rank_agreement import raise_on_every_rank, run_on_every_rank
from .device import DeviceLayout
from .host_words import host_module, key_rows
from .plan import LayoutPlan, agree
from .tile_dispatch import ENGINES

MAX_KEY_WORDS = 2

DEFAULT_NUM_SLICES = 4
logger = logging.getLogger(__name__)


class FullStrsGPUHamiltonian:
    """GPU-backed real type-2 Hamiltonian on an explicit set of full determinants.

    Use with ``distributed_primme_eigsh`` to solve eigenvalue problems on one or
    more ranks. Full determinants must use block spin ordering, with alpha in the
    low bits and beta in the high bits, equal sector widths of 1–128 orbitals,
    fixed sector electron counts and supported spin-conserving terms.
    Unsupported operator groups are rejected.

    ``keys`` is a uint64 array of shape ``(n, 2W)``. W is 1 for up to 64
    orbitals per spin sector and 2 for 65 to 128. Each key holds the W alpha
    words, then the W beta words, low word first. Bit k of word w is set when
    orbital 64w + k is occupied. The determinants need not form a complete
    tensor-product subspace.

    The keys can be unsorted and can contain duplicates. The constructor
    removes duplicates, sorts the keys and distributes them for the selected
    layout. After construction, ``keys`` holds this rank's keys in their final
    order, ``n_local`` is the number of determinants this rank owns, and
    ``row_start`` is the global index of its first determinant. Local vectors
    and ``diag_local`` use the same order: entry i belongs to ``keys[i]``.

    Args:
        operator: Real type-2 ``QubitOperator`` in the block layout above.
        keys: This rank's input determinants, uint64 shape ``(n, 2W)``.
            Input assignment and order may change during construction.
        comm: MPI or ``LocalComm`` communicator for all participating ranks.
            ``None`` uses ``MPI.COMM_SELF``.
        engine: ``"auto"`` (default), ``"tile"`` or ``"general"``.
            ``"tile"`` uses the faster ``TileGPUResidentHamiltonian`` and
            needs a Cartesian-product subspace: the keys must hold every
            alpha string paired with every beta string. ``"general"`` works
            for any set of keys. ``"auto"`` uses ``"tile"`` when it can, and
            ``"general"`` otherwise. ``"tile"`` raises an error when it cannot
            run. The ``engine`` attribute shows the engine in use. Both
            engines use the same key order. For only a few matvecs, use
            ``"general"``: the tile engine can take longer to build at high
            orbital counts.

    Raises:
        ValueError: Keys, operator terms or the engine are unsupported.
        ImportError: CuPy or required native components are unavailable.
        MemoryError: GPU allocations cannot fit in available device memory.
        RuntimeError: Device or communication setup fails.
    """

    _DEFAULT_OPTIONS = {
        "num_slices": DEFAULT_NUM_SLICES,
        "binned": True,
        "residency": "auto",
        "device_budget_bytes": None,
        "alpha_kind": None,
        "beta_kind": None,
        "recut": None,
        "exchange": "stepped",
        "copies": 1,
        "plan": None,
        "fixed_reduce": True,
        "release_host_copies": True,
        "domain_size": None,
        "groups": "by_rows",
        "atomic_reduce": False,
        "vectors": "halo",
    }

    def __init__(self, operator, keys, *, comm=None, engine="auto", _options=None):
        started = time.perf_counter()
        options = dict(self._DEFAULT_OPTIONS)
        unknown = sorted(set(_options or {}) - set(options))
        if unknown:
            raise ValueError(f"unknown options: {', '.join(unknown)}")
        options.update(_options or {})

        num_slices = options["num_slices"]
        binned = options["binned"]
        residency = options["residency"]
        device_budget_bytes = options["device_budget_bytes"]
        alpha_kind = options["alpha_kind"]
        beta_kind = options["beta_kind"]
        recut = options["recut"]
        exchange = options["exchange"]
        copies = options["copies"]
        plan = options["plan"]
        fixed_reduce = options["fixed_reduce"]
        release_host_copies = options["release_host_copies"]
        domain_size = options["domain_size"]
        groups = options["groups"]
        atomic_reduce = options["atomic_reduce"]
        vectors = options["vectors"]

        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        if comm is None:
            from mpi4py import MPI  # noqa: PLC0415

            comm = MPI.COMM_SELF
        self._comm = comm
        num_orbitals, electrons = run_on_every_rank(
            comm, "operator width", lambda: _orbitals_and_electrons(operator, keys)
        )
        check_width(
            comm,
            (num_orbitals, num_orbitals),
            "FullStrsGPUHamiltonian",
            electrons,
            max_words=MAX_KEY_WORDS,
        )
        keys = run_on_every_rank(
            comm, "input keys", lambda: key_rows(keys, num_orbitals)
        )
        self._key_words = np.shape(keys)[1] // 2
        self._tile = None
        self._block = None
        self._copies = None
        self._full = None
        self.engine = "general"
        defaults = (
            num_slices == DEFAULT_NUM_SLICES
            and binned is True
            and residency == "auto"
            and device_budget_bytes is None
            and alpha_kind is None
            and beta_kind is None
            and recut is None
            and exchange == "stepped"
            and copies == 1
            and plan is None
            and fixed_reduce is True
            and release_host_copies is True
            and domain_size is None
            and groups == "by_rows"
            and atomic_reduce is False
            and vectors == "halo"
        )
        if engine == "tile" and not defaults:
            raise ValueError(
                "engine='tile' takes no general-engine options; leave them at their defaults"
            )
        if (
            engine != "general"
            and defaults
            and self._init_tile(operator, keys, num_orbitals, engine, started)
        ):
            return
        self._binned = bool(binned)
        self._block = None
        self._copies = None
        self._fixed_reduce = bool(fixed_reduce)
        self._release_host_copies = bool(release_host_copies)
        self._domain_size = domain_size
        if groups not in ("by_rows", "by_key"):
            raise ValueError(f"groups must be 'by_rows' or 'by_key', got {groups!r}")
        self._groups = groups
        if vectors not in ("halo", "full") or (
            vectors == "full" and groups != "by_key"
        ):
            raise ValueError(
                f"vectors must be 'halo', or 'full' with groups='by_key'; got {vectors!r}"
            )
        self._vectors = vectors
        self._full = None
        self._atomic_reduce = bool(atomic_reduce) or vectors == "full"
        self._y_ext = None
        if plan is not None:
            plan = LayoutPlan.from_dict(
                plan if isinstance(plan, dict) else plan.to_dict()
            )
            plan.check()
            agree(comm, plan)
            copies, num_slices, exchange = plan.copies, plan.num_slices, plan.exchange
            recut = plan.block_cut != "row" and not plan.fixed
        auto_candidates = None
        if copies == "auto" and comm.Get_size() == 1:
            copies = 1
        if copies == "auto":
            from .distribute import plan_copies  # noqa: PLC0415

            copies, auto_candidates = plan_copies(
                comm,
                keys,
                _orbitals(operator),
                int(num_slices),
                alpha_kind=alpha_kind,
                beta_kind=beta_kind,
            )
        copies = int(copies)
        if exchange == "two_level" and copies > 1:
            raise ValueError(
                "the two-level exchange supports one copy per block; use copies=1"
            )
        if groups == "by_key" and copies > 1:
            raise ValueError(
                "groups='by_key' supports one copy per block; use copies=1"
            )
        if groups == "by_key" and plan is None:
            recut = False
        if plan is None:
            if recut is None and comm.Get_size() > 1:
                plan = LayoutPlan(
                    copies=copies,
                    num_slices=int(num_slices),
                    exchange=exchange,
                    block_cut="fitted",
                    shares="balanced",
                    coefficients=[1.0] * 35,
                )
            else:
                plan = LayoutPlan(
                    copies=copies,
                    num_slices=int(num_slices),
                    exchange=exchange,
                    block_cut="membership" if recut and comm.Get_size() > 1 else "row",
                )
            plan.check()
        recut = plan.block_cut != "row" and not plan.fixed
        if auto_candidates is not None:
            plan.measured["copies_candidates"] = {
                str(t): v for t, v in auto_candidates.items()
            }
        self.plan = plan
        if copies > 1:
            self._init_copies(
                operator,
                keys,
                int(num_slices),
                residency,
                device_budget_bytes,
                alpha_kind,
                beta_kind,
                copies,
                exchange,
            )
            agree(comm, self.plan)
            self._log_ready(started)
            return
        if comm.Get_size() > 1:
            self._init_ranks(
                operator,
                keys,
                int(num_slices),
                residency,
                device_budget_bytes,
                alpha_kind,
                beta_kind,
                bool(recut),
                exchange,
            )
            agree(comm, self.plan)
            self._log_ready(started)
            return
        self._is_primary = True
        self.row_start = 0
        self._host = host_module(self._key_words).HostOperator(
            operator, keys, alpha_kind, beta_kind, num_slices=int(num_slices)
        )
        self._layout = DeviceLayout(
            self._host,
            residency,
            device_budget_bytes,
            self._fixed_reduce,
            atomic_reduce=self._atomic_reduce,
        )
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = int(self._host.num_rows)
        self.plan.share_starts = [0, self.n_local]
        self.shape = (self.n_local, self.n_local)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self.diag_local = np.array(self._host.diagonal_view(), dtype=np.float64)
        self._log_ready(started)

    def _init_tile(self, operator, keys, num_orbitals, engine, started):

        from ..distributed.dist_matvec_tile_resident import (  # noqa: PLC0415
            TileGPUResidentHamiltonian,
        )
        from .distribute import distribute_keys, redistribute  # noqa: PLC0415
        from .tile_dispatch import beta_cuts, prepare  # noqa: PLC0415

        comm = self._comm
        block = distribute_keys(comm, keys, num_orbitals)
        prepared, reason = prepare(comm, operator, block, self._key_words, num_orbitals)
        if prepared is None:
            if engine == "tile":
                raise ValueError(f"engine='tile' cannot run: {reason}")
            if comm.Get_rank() == 0:
                logger.debug(
                    "Full-string GPU Hamiltonian uses the general engine: %s",
                    reason,
                    extra={"gpu_rank": 0},
                )
            return False
        hsub, alpha, beta = prepared
        block = redistribute(
            comm, block, beta_cuts(len(alpha), len(beta), comm.Get_size())
        )
        self._tile = TileGPUResidentHamiltonian(
            hsub,
            alpha,
            beta,
            comm=comm,
            process_grid=(1, comm.Get_size()),
            gpu_ids=(int(cupy.cuda.runtime.getDevice()),),
        )
        tile = self._tile
        raise_on_every_rank(
            comm,
            None
            if tile.n_local == block.n_local
            else f"the tile holds {tile.n_local} rows, the key block {block.n_local}",
            RuntimeError,
            "tile engine layout",
        )
        self.engine = "tile"
        self._is_primary = True
        self._tile_keys = block.keys
        self._host = _TileHost(num_orbitals)
        self.n_local = block.n_local
        self.row_start = block.row_start
        self.shape = (block.num_rows, block.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self.diag_local = np.array(tile.diag_local, dtype=np.float64)
        self.plan = LayoutPlan(copies=1, num_slices=DEFAULT_NUM_SLICES)
        self.plan.share_starts = [int(v) for v in block.block_starts]
        self.plan.measured["engine"] = "tile"
        if comm.Get_rank() == 0:
            logger.debug(
                "Full-string GPU Hamiltonian uses the tile engine: grid=1x%d alpha=%d beta=%d",
                comm.Get_size(),
                len(alpha),
                len(beta),
                extra={"gpu_rank": 0},
            )
        self._log_ready(started)
        return True

    def _log_ready(self, started):

        if self._comm.Get_rank() == 0:
            logger.info(
                "Full-string GPU Hamiltonian ready: ranks=%d, rank0_vector_elements=%d, construction=%.3f s",
                self._comm.Get_size(),
                self.n_local,
                time.perf_counter() - started,
                extra={"gpu_rank": 0},
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Full-string GPU layout: %s; device storage (bytes): %s",
                    self.plan.to_dict(),
                    self.device_bytes,
                    extra={"gpu_rank": 0},
                )

    def _init_ranks(
        self,
        operator,
        keys,
        num_slices,
        residency,
        device_budget_bytes,
        alpha_kind,
        beta_kind,
        recut,
        exchange,
    ):

        from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

        from .distribute import build_distributed_groups  # noqa: PLC0415
        from .halo import HaloExchange  # noqa: PLC0415

        host = host_module(self._key_words)
        RankOperator, block_reference = host.RankOperator, host.block_reference

        comm = self._comm
        num_orbitals = run_on_every_rank(
            comm, "operator width", lambda: _orbitals(operator)
        )
        coefficients = None
        if self.plan.block_cut == "fitted":
            coefficients = (
                self.plan.coefficients
                if self.plan.coefficients is not None
                else [1.0] * 35
            )
        block, receiver, self.recut_info = build_distributed_groups(
            comm,
            keys,
            num_orbitals,
            num_slices,
            alpha_kind,
            beta_kind,
            recut,
            self.plan.share_starts,
            coefficients,
            self.plan.entry_cost,
            fetch_keys=exchange != "two_level",
            groups=self._groups,
            vectors=self._vectors,
        )
        self.plan.measured["groups"] = self._groups
        self.plan.measured["vectors"] = self._vectors
        if self._vectors == "full":
            self._init_full(
                operator, block, receiver, num_orbitals, residency, device_budget_bytes
            )
            return
        self.plan.share_starts = [int(v) for v in block.block_starts]
        self.plan.measured.update(
            {
                "imbalance_before": self.recut_info["imbalance_before"],
                "imbalance_after": self.recut_info["imbalance_after"],
            }
        )
        domains = None
        if exchange == "two_level":
            from .distribute import fetch_halo_keys  # noqa: PLC0415
            from .two_level import exchange_domains, extend_to_domain_union  # noqa: PLC0415

            domain_comm, domains = exchange_domains(comm, self._domain_size)
            extend_to_domain_union(comm, domain_comm, domains, block, receiver)
            fetch_halo_keys(comm, block, receiver)
        reference = comm.bcast(
            block_reference(block.keys, num_orbitals) if comm.Get_rank() == 0 else None,
            root=0,
        )
        rank_host = run_on_every_rank(
            comm, "table build", lambda: RankOperator(operator, block.keys, reference)
        )
        self._block = block
        self._receiver = receiver
        self._host = (_KeyHost if self._groups == "by_key" else _RankHost)(
            rank_host, receiver, num_orbitals
        )

        if exchange == "two_level":
            from .two_level import TwoLevelExchange  # noqa: PLC0415

            self._halo = TwoLevelExchange(
                comm, block, receiver.halo_row_view(), domains
            )
        else:
            self._halo = HaloExchange(comm, block, receiver.halo_row_view(), exchange)
        tail = getattr(self._halo, "tail_elements", 0)

        y_elements = (
            self._host.num_rows + self._halo.reverse_tail_elements
            if self._groups == "by_key"
            else 0
        )
        reserve = tail + max(0, y_elements - self._host.num_own)
        self._layout = run_on_every_rank(
            comm,
            "device layout",
            lambda: DeviceLayout(
                self._host,
                residency,
                device_budget_bytes,
                self._fixed_reduce,
                reserve_elements=reserve,
                atomic_reduce=self._atomic_reduce,
            ),
        )
        if self._release_host_copies:
            self._layout.release_host_chunks()
            receiver.release_halo()
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = block.n_local
        self.row_start = block.row_start
        total = sum(comm.allgather(self.n_local))
        if total != block.num_rows:
            raise ValueError(
                f"the row blocks hold {total} rows, expected {block.num_rows}"
            )
        self.shape = (block.num_rows, block.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self._is_primary = True
        self.diag_local = np.array(rank_host.diagonal_view(), dtype=np.float64)

        self._x_ext = cupy.zeros(self._host.num_rows + tail, dtype=np.float64)
        if y_elements:
            self._y_ext = cupy.zeros(y_elements, dtype=np.float64)
        cupy.cuda.Device().synchronize()

    def _init_full(
        self, operator, block, receiver, num_orbitals, residency, device_budget_bytes
    ):

        from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

        from .full_vectors import FullVectorExchange  # noqa: PLC0415

        host = host_module(self._key_words)
        RankOperator, block_reference = host.RankOperator, host.block_reference

        comm = self._comm
        size, rank = comm.Get_size(), comm.Get_rank()
        stride = int(np.diff(block.block_starts).max())
        reference = comm.bcast(
            block_reference(block.keys, num_orbitals) if rank == 0 else None, root=0
        )
        rank_host = run_on_every_rank(
            comm, "table build", lambda: RankOperator(operator, block.keys, reference)
        )
        exchange = FullVectorExchange(comm, stride, block.n_local)
        stream = cupy.cuda.Stream(non_blocking=True)

        row_words = 2 * self._key_words
        with stream:
            keys = cupy.zeros(row_words * size * stride, dtype=np.uint64)
        first = row_words * rank * stride
        keys[first : first + row_words * block.n_local].set(
            np.ascontiguousarray(block.keys).reshape(-1), stream=stream
        )
        exchange.allgather(keys.view(np.float64), stream, width=row_words)
        stream.synchronize()
        self._block = block
        self._receiver = receiver
        self._host = _FullHost(rank_host, receiver, num_orbitals, size * stride)
        self._halo = exchange
        reserve = self._host.num_rows - self._host.num_own
        self._layout = run_on_every_rank(
            comm,
            "device layout",
            lambda: DeviceLayout(
                self._host,
                residency,
                device_budget_bytes,
                False,
                reserve_elements=reserve,
                atomic_reduce=True,
                device_keys=keys,
            ),
        )
        if self._release_host_copies:
            self._layout.release_host_chunks()
        self._full = (exchange, stride)
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = block.n_local
        self.row_start = block.row_start
        total = sum(comm.allgather(self.n_local))
        if total != block.num_rows:
            raise ValueError(
                f"the row blocks hold {total} rows, expected {block.num_rows}"
            )
        self.shape = (block.num_rows, block.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self._is_primary = True
        self.diag_local = np.array(rank_host.diagonal_view(), dtype=np.float64)
        self._x_ext = cupy.zeros(size * stride, dtype=np.float64)
        self._y_ext = cupy.zeros(size * stride, dtype=np.float64)
        cupy.cuda.Device().synchronize()

    def _init_copies(
        self,
        operator,
        keys,
        num_slices,
        residency,
        device_budget_bytes,
        alpha_kind,
        beta_kind,
        copies,
        exchange,
    ):

        from fulqrum.gpu.distributed.rank_agreement import run_on_every_rank  # noqa: PLC0415

        from .distribute import plan_copy_groups  # noqa: PLC0415
        from .halo import CopyExchange, HaloExchange  # noqa: PLC0415

        host = host_module(self._key_words)
        RankOperator, block_reference = host.RankOperator, host.block_reference

        comm = self._comm
        num_orbitals = run_on_every_rank(
            comm, "operator width", lambda: _orbitals(operator)
        )
        layout, measured = plan_copy_groups(
            comm, keys, num_orbitals, self.plan, alpha_kind, beta_kind
        )
        self.plan.measured.update(measured)
        reference = comm.bcast(
            block_reference(layout.block_keys, num_orbitals)
            if comm.Get_rank() == 0
            else None,
            root=0,
        )
        rank_host = run_on_every_rank(
            comm,
            "table build",
            lambda: RankOperator(operator, layout.block_keys, reference),
        )
        share = layout.share
        share_begin = share.row_start - layout.block_start
        self._copies = layout
        self._block = share
        self._host = _CopyHost(
            rank_host,
            layout.pieces,
            num_orbitals,
            share_begin,
            share_begin + share.n_local,
        )
        self._layout = run_on_every_rank(
            comm,
            "device layout",
            lambda: DeviceLayout(
                self._host,
                residency,
                device_budget_bytes,
                self._fixed_reduce,
                atomic_reduce=self._atomic_reduce,
            ),
        )
        if self._release_host_copies:
            self._layout.release_host_chunks()
        self._halo = HaloExchange(comm, share, layout.pieces.halo_rows(), exchange)
        if self._release_host_copies:
            layout.pieces.release_halo()
        share_starts = np.array(
            layout.copy_comm.allgather(share_begin) + [len(layout.block_keys)],
            dtype=np.int64,
        )
        self._copy_exchange = CopyExchange(layout.copy_comm, share_starts)
        self._stream = cupy.cuda.Stream(non_blocking=True)
        self._layout.wait(self._stream)
        self.n_local = share.n_local
        self.row_start = share.row_start
        total = sum(comm.allgather(self.n_local))
        if total != share.num_rows:
            raise ValueError(f"the shares hold {total} rows, expected {share.num_rows}")
        self.shape = (share.num_rows, share.num_rows)
        self.dtype = np.dtype(np.float64)
        self.supports_device_matvec = True
        self._is_primary = True
        self.diag_local = np.array(
            rank_host.diagonal_view()[share_begin : share_begin + share.n_local],
            dtype=np.float64,
        )
        self.plan.share_starts = [int(v) for v in share.block_starts]
        self.plan.assignment = layout.plan["assignment"].tolist()
        self.plan.measured["work"] = layout.plan["work"].tolist()
        self._x_ext = cupy.zeros(self._host.num_rows, dtype=np.float64)
        self._y_block = cupy.zeros(self._host.num_own, dtype=np.float64)

    @property
    def keys(self):
        """Return this rank's post-distribution keys as a uint64 ``(n_local, 2W)`` copy, alpha
        words then beta words, sorted by ``(beta, alpha)``."""
        if self._tile is not None:
            return self._tile_keys.copy()
        if self._block is not None:
            return self._block.keys.copy()
        return self._host.sorted_keys()

    @property
    def num_orbitals(self):
        """The number of orbitals per spin sector."""
        return self._host.num_orbitals

    @property
    def num_halo(self):
        """The number of remote rows that this rank reads (0 with one rank)."""
        return 0 if self._block is None else self._halo.num_halo

    @property
    def num_streamed_slices(self):
        """The number of key slices streamed from host memory at every matvec."""
        return 0 if self._tile is not None else self._layout.num_streamed

    @property
    def device_bytes(self):
        """The device memory of the stored groups, tables, keys, diagonal and matvec buffers, in bytes."""
        if self._tile is not None:
            return dict(self._tile.predicted_hbm_bytes)
        return self._layout.device_bytes()

    def _matvec(self, x, out=None):

        if self._tile is not None:
            return self._tile._matvec(x, out=out)
        if out is None:
            out = cupy.empty(self.n_local, dtype=np.float64)
        caller = cupy.cuda.get_current_stream()
        self._stream.wait_event(caller.record())
        if self._block is None:
            self._layout.matvec(x, out, self._stream, self._binned)
        elif self._copies is not None:
            block = self._host.num_own
            self._copy_exchange.allgather(x, self._x_ext[:block], self._stream)
            self._halo.run(x, self._x_ext[block:], self._stream)
            self._layout.matvec(self._x_ext, self._y_block, self._stream, self._binned)
            self._copy_exchange.reduce_scatter(self._y_block, out, self._stream)
        elif self._full is not None:
            exchange, stride = self._full
            own = self.n_local
            start = self._comm.Get_rank() * stride
            self._x_ext[start : start + own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            exchange.allgather(self._x_ext, self._stream)
            with self._stream:
                self._y_ext.fill(0.0)
            self._layout.matvec(
                self._x_ext, self._y_ext, self._stream, self._binned, diagonal=False
            )
            exchange.reduce_scatter(self._y_ext, self._stream)
            with self._stream:
                cupy.add(
                    cupy.multiply(self._layout.diagonal, x),
                    self._y_ext[start : start + own],
                    out=out,
                )
        elif self._y_ext is not None:
            own, rows = self.n_local, self._host.num_rows
            self._x_ext[:own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            self._halo.run(self._x_ext[:own], self._x_ext[own:], self._stream)
            with self._stream:
                self._y_ext[own:rows].fill(0.0)
            self._layout.matvec(self._x_ext, self._y_ext, self._stream, self._binned)
            self._halo.reverse(self._y_ext[:own], self._y_ext[own:], self._stream)
            out.data.copy_from_device_async(
                self._y_ext.data, 8 * own, stream=self._stream
            )
        else:
            own = self.n_local
            self._x_ext[:own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            self._halo.run(self._x_ext[:own], self._x_ext[own:], self._stream)
            self._layout.matvec(self._x_ext, out, self._stream, self._binned)
        caller.wait_event(self._stream.record())
        return out

    def _rdm_input(self, x):

        if self._block is None:
            return x
        if self._copies is not None:
            block = self._host.num_own
            self._copy_exchange.allgather(x, self._x_ext[:block], self._stream)
            self._halo.run(x, self._x_ext[block:], self._stream)
            return self._x_ext
        own = self.n_local
        if self._full is not None:
            exchange, stride = self._full
            start = self._comm.Get_rank() * stride
            self._x_ext[start : start + own].data.copy_from_device_async(
                x.data, 8 * own, stream=self._stream
            )
            exchange.allgather(self._x_ext, self._stream)
            return self._x_ext
        self._x_ext[:own].data.copy_from_device_async(
            x.data, 8 * own, stream=self._stream
        )
        self._halo.run(self._x_ext[:own], self._x_ext[own:], self._stream)
        return self._x_ext

    def _rdm_own_keys(self):

        if self._copies is not None:
            first = self._host.share[0]
        elif self._full is not None:
            first = self._comm.Get_rank() * self._full[1]
        else:
            first = 0
        row_words = 2 * self._key_words
        return self._layout.keys[row_words * first : row_words * (first + self.n_local)]

    def compute_rdms(self, x, *, rank=1, spin_summed=False):
        """Collectively compute RDMs from this rank's NumPy or CuPy state slice, shape
        ``(n_local,)`` in ``keys`` order. Every rank receives full host NumPy arrays after
        device work and collectives complete; ``rank`` selects the RDM order (1 or 2), and no
        normalization is applied. Spin blocks, spin-summed forms, and conventions follow PySCF
        ``make_rdm12s(reorder=True)``."""
        if self._tile is not None:
            return self._tile.compute_rdms(x, rank=rank, spin_summed=spin_summed)
        from .rdm import compute_rdms_full_strs  # noqa: PLC0415

        return compute_rdms_full_strs(self, x, rank=rank, spin_summed=spin_summed)

    def matvec(self, x):
        """Collectively apply the operator to this rank's local vector of shape ``(n_local,)`` in
        ``keys`` order. NumPy input returns NumPy output; CuPy input returns CuPy output."""
        if tuple(x.shape) != (self.n_local,):
            raise ValueError(
                f"x must have shape ({self.n_local},); got {tuple(x.shape)}"
            )
        if isinstance(x, cupy.ndarray):
            return self._matvec(cupy.ascontiguousarray(x, dtype=np.float64))
        d_x = cupy.asarray(np.ascontiguousarray(x, dtype=np.float64))
        y = self._matvec(d_x)
        return cupy.asnumpy(y)


def _key_electrons(keys):

    keys = np.asarray(keys)
    if keys.ndim != 2 or keys.shape[0] == 0 or keys.shape[1] % 2:
        return (None, None)
    half = keys.shape[1] // 2
    words = np.ascontiguousarray(keys, dtype=np.uint64)
    return (
        int(popcount_w(words[:, :half]).max()),
        int(popcount_w(words[:, half:]).max()),
    )


def _orbitals_and_electrons(operator, keys):

    return _orbitals(operator), _key_electrons(keys)


def _orbitals(operator):

    width = int(operator.width)
    if width <= 0 or width % 2:
        raise ValueError(
            f"the operator must have an even, positive qubit count, got {width}"
        )
    return width // 2


class _TileHost:
    def __init__(self, num_orbitals):

        self.num_orbitals = int(num_orbitals)


class _RankHost:
    def __init__(self, rank_host, receiver, num_orbitals):

        self.rank_host = rank_host
        self.receiver = receiver
        self.num_orbitals = num_orbitals
        self.num_own = rank_host.num_own
        self.num_rows = self.num_own + receiver.halo_row_view().size
        self.num_chunks = receiver.num_chunks()

    def key_words(self):

        return self.rank_host.key_words()

    def halo_key_words(self):

        return self.receiver.halo_key_words()

    def diagonal_view(self):

        return self.rank_host.diagonal_view()

    def table_views(self):

        return self.rank_host.table_views()

    def chunk_views(self, chunk, cls):

        return self.receiver.chunk_views(chunk, cls)

    def release_units(self, units, num_kinds):

        for unit in units:
            self.receiver.release_chunk(unit // num_kinds)


class _KeyHost(_RankHost):
    def chunk_kinds(self):

        return [(0, self.num_rows, 0, self.num_rows, 0, self.num_rows)]

    def chunk_views(self, chunk, cls, kind=0):

        return self.receiver.chunk_views(chunk, cls)


class _FullHost(_RankHost):
    def __init__(self, rank_host, receiver, num_orbitals, num_rows):

        super().__init__(rank_host, receiver, num_orbitals)
        self.num_rows = int(num_rows)

    def chunk_kinds(self):

        return [(0, self.num_rows, 0, self.num_rows, 0, self.num_rows)]

    def chunk_views(self, chunk, cls, kind=0):

        return self.receiver.chunk_views(chunk, cls)


class _CopyHost:
    def __init__(self, rank_host, pieces, num_orbitals, share_begin, share_end):

        self.rank_host = rank_host
        self.pieces = pieces
        self.num_orbitals = num_orbitals
        self.num_own = pieces.num_own
        self.num_rows = self.num_own + pieces.halo_rows().size
        self.num_chunks = pieces.num_chunks()
        self.share = (int(share_begin), int(share_end))
        diagonal = np.zeros(self.num_own, dtype=np.float64)
        diagonal[share_begin:share_end] = rank_host.diagonal_view()[
            share_begin:share_end
        ]
        self.diagonal = diagonal

    def chunk_kinds(self):

        own = tuple(self.pieces.ranges(0)) + self.share
        remote = tuple(self.pieces.ranges(1)) + (0, self.num_own)
        return [own, remote]

    def chunk_views(self, chunk, cls, kind):

        return self.pieces.chunk_views(chunk, cls, kind)

    def release_units(self, units, num_kinds):

        for unit in units:
            self.pieces.release_chunk(unit // num_kinds, unit % num_kinds)

    def key_words(self):

        return self.rank_host.key_words()

    def halo_key_words(self):

        return self.pieces.halo_key_words()

    def diagonal_view(self):

        return self.diagonal

    def table_views(self):

        return self.rank_host.table_views()
