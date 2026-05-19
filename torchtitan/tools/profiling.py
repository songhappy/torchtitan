# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000

# Pattern to strip layer/module indices from op names for grouping.
# Matches things like "(layers.4)", "(output)", "(norm)", "(layers.30.attn)", etc.
_LAYER_SUFFIX_RE = re.compile(r"\s*\((?:layers\.\d+[^)]*|[a-z_]+)\)\s*$")


def _build_grouped_summary(key_averages, sort_by: str, row_limit: int = 50) -> str:
    """
    Build a summary table where ops are grouped by base name.

    E.g. all 'FSDP::post_backward_reduce (layers.N)' entries become one
    'FSDP::post_backward_reduce' row with summed times and counts.
    """
    groups: dict[str, dict] = defaultdict(
        lambda: {
            "cpu_time_total": 0.0,
            "device_time_total": 0.0,
            "self_cpu_time_total": 0.0,
            "self_device_time_total": 0.0,
            "count": 0,
        }
    )

    for evt in key_averages:
        base_name = _LAYER_SUFFIX_RE.sub("", evt.key)
        g = groups[base_name]
        g["cpu_time_total"] += getattr(evt, "cpu_time_total", 0)
        g["self_cpu_time_total"] += getattr(evt, "self_cpu_time_total", 0)
        g["count"] += evt.count
        # Try device_time_total first (newer PyTorch), then cuda/xpu variants
        device_total = (
            getattr(evt, "device_time_total", 0)
            or getattr(evt, "cuda_time_total", 0)
            or getattr(evt, "xpu_time_total", 0)
        )
        self_device_total = (
            getattr(evt, "self_device_time_total", 0)
            or getattr(evt, "self_cuda_time_total", 0)
            or getattr(evt, "self_xpu_time_total", 0)
        )
        g["device_time_total"] += device_total
        g["self_device_time_total"] += self_device_total

    # Sort by device time or cpu time
    if "cuda" in sort_by or "xpu" in sort_by:
        sort_col = "device_time_total"
    else:
        sort_col = sort_by
    rows = sorted(groups.items(), key=lambda x: x[1].get(sort_col, 0), reverse=True)
    rows = rows[:row_limit]

    # Build table
    use_device = "cuda" in sort_by or "xpu" in sort_by
    header_device = sort_by.replace("_time_total", "").upper()

    lines = []
    sep = "-" * 60 + "  " + "  ".join(["-" * 12] * (5 if use_device else 3))
    if use_device:
        header = (
            f"{'Name':>60s}  {'Self CPU':>12s}  {'CPU total':>12s}  "
            f"{'Self ' + header_device:>12s}  {header_device + ' total':>12s}  {'# Calls':>12s}"
        )
    else:
        header = (
            f"{'Name':>60s}  {'Self CPU':>12s}  {'CPU total':>12s}  {'# Calls':>12s}"
        )
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    def fmt_us(us: float) -> str:
        if us >= 1e6:
            return f"{us / 1e6:.3f}s"
        elif us >= 1e3:
            return f"{us / 1e3:.3f}ms"
        else:
            return f"{us:.3f}us"

    for name, g in rows:
        display_name = name if len(name) <= 60 else name[:57] + "..."
        if use_device:
            lines.append(
                f"{display_name:>60s}  {fmt_us(g['self_cpu_time_total']):>12s}  "
                f"{fmt_us(g['cpu_time_total']):>12s}  "
                f"{fmt_us(g['self_device_time_total']):>12s}  "
                f"{fmt_us(g['device_time_total']):>12s}  "
                f"{g['count']:>12d}"
            )
        else:
            lines.append(
                f"{display_name:>60s}  {fmt_us(g['self_cpu_time_total']):>12s}  "
                f"{fmt_us(g['cpu_time_total']):>12s}  "
                f"{g['count']:>12d}"
            )
    lines.append(sep)
    return "\n".join(lines)


# TODO: introduce an owner class, namely Profiler
@dataclass(kw_only=True, slots=True)
class ProfilingConfig:
    enable_profiling: bool = False
    """Whether to enable pytorch profile"""

    save_traces_folder: str = "profile_traces"
    """Trace files location"""

    profile_freq: int = 10
    """How often to collect profile traces, in iterations"""

    profiler_active: int = 1
    """
    The steps profiler is active for.

    This is used to configure torch.profile.schedule.
    """

    profiler_warmup: int = 3
    """
    The number of warmup steps before the active step in each profiling cycle.

    This is used to configure torch.profile.schedule.
    """

    profiler_repeat: int | None = None
    """
    The number of times to repeat the profiling cycle

    This is used to configure torch.profile.schedule.
    """

    profiler_skip_first: int | None = None
    """
    The number of initial profiling cycles to skip

    This is used to configure torch.profile.schedule.
    """

    profiler_skip_first_wait: int | None = None
    """
    The number of initial profiling cycles to skip the wait time

    This is used to configure torch.profile.schedule.
    """

    enable_memory_snapshot: bool = False
    """Whether to dump memory snapshot"""

    save_memory_snapshot_folder: str = "memory_snapshot"
    """Memory snapshot files location"""


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    # get user defined profiler settings
    enable_profiling = profiling_config.enable_profiling

    if enable_profiling:
        trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
        profile_freq, warmup, active = (
            profiling_config.profile_freq,
            profiling_config.profiler_warmup,
            profiling_config.profiler_active,
        )

        additional_params = {
            key: val
            for key, val in [
                ("repeat", profiling_config.profiler_repeat),
                ("skip_first", profiling_config.profiler_skip_first),
                ("skip_first_wait", profiling_config.profiler_skip_first_wait),
            ]
            if val is not None
        }

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
            prof.export_chrome_trace(output_file)
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

            # Save profiler key averages table (rank 0 only)
            if rank == 0:
                key_averages = prof.key_averages()
                tables = []
                grouped_tables = []
                if torch.cuda.is_available():
                    sort_key = "cuda_time_total"
                    tables.append(
                        key_averages.table(
                            sort_by=sort_key,
                            max_name_column_width=60,
                            row_limit=100,
                        )
                    )
                    grouped_tables.append(
                        _build_grouped_summary(key_averages, sort_by=sort_key)
                    )
                elif torch.xpu.is_available():
                    sort_key = "xpu_time_total"
                    tables.append(
                        key_averages.table(
                            sort_by=sort_key,
                            max_name_column_width=60,
                            row_limit=100,
                        )
                    )
                    grouped_tables.append(
                        _build_grouped_summary(key_averages, sort_by=sort_key)
                    )
                if len(key_averages) > 0 and hasattr(key_averages[0], "cpu_time_total"):
                    tables.append(
                        key_averages.table(
                            sort_by="cpu_time_total",
                            max_name_column_width=60,
                            row_limit=100,
                        )
                    )
                    grouped_tables.append(
                        _build_grouped_summary(
                            key_averages, sort_by="cpu_time_total"
                        )
                    )
                # Print detailed tables to stdout
                for t in tables:
                    print(t)
                # Print grouped summary
                print("\n=== Grouped Summary (ops aggregated across layers) ===")
                for t in grouped_tables:
                    print(t)

                # Save detailed tables
                summary_file = os.path.join(curr_trace_dir, "profiler_key_averages.txt")
                with open(summary_file, "w") as f:
                    for t in tables:
                        f.write(t)
                        f.write("\n")
                # Save grouped summary
                grouped_file = os.path.join(
                    curr_trace_dir, "profiler_key_averages_grouped.txt"
                )
                with open(grouped_file, "w") as f:
                    for t in grouped_tables:
                        f.write(t)
                        f.write("\n")
                logger.info(
                    f"Profiler key averages saved to {summary_file} "
                    f"and grouped summary to {grouped_file}"
                )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            # pyrefly: ignore [bad-argument-type]
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, **additional_params
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    enable_snapshot = profiling_config.enable_memory_snapshot
    if enable_snapshot:
        snapshot_dir = os.path.join(
            base_folder, profiling_config.save_memory_snapshot_folder
        )
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                device_module.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name, leaf_folder)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                output_file = os.path.join(
                    curr_snapshot_dir, f"rank{rank}_memory_snapshot.pickle"
                )
                with open(output_file, "wb") as output:
                    pickle.dump(device_module.memory._snapshot(), output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, profiling_config.profile_freq)
        try:
            yield profiler
        except torch.OutOfMemoryError:
            profiler.step(exit_ctx=True)
            raise
    else:
        yield None
