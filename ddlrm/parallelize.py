import os
import copy
import time
import random
import inspect
from typing import Any, Callable, Dict, Optional
import torch
from functools import partial
from torch.fx.passes import graph_drawer
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
import tracy_client as tracy


def parallel_embedding_bag(
    weight: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    scale_grad_by_freq: bool = False,
    mode: int = 0,  # 0: sum, 1: mean, 2: max
    sparse: bool = False,
    per_sample_weights: torch.Tensor = None,
    include_last_offset: bool = False,
    padding_idx: int = -1,
    split_mode: str = 'horizontal',  # 'horizontal' or 'vertical'
    num_chunks: int = 2  # number of chunks for parallelism
):

    # Define the mode mapping for F.embedding_bag
    mode_map = {0: 'sum', 1: 'mean', 2: 'max'}
    reduction_mode = mode_map.get(mode, 'sum')

    # Horizontal Splitting: Split input indices across the batch dimension
    def parallel_embedding_bag_horizontal():
        split_indices = torch.chunk(indices, num_chunks, dim=0)
        split_offsets = torch.chunk(offsets, num_chunks, dim=0)

        # Function to compute embedding bag for each chunk

        def embedding_bag_chunk(args):
            idx, off = args
            return torch.ops.aten._embedding_bag(
                weight, idx, off,
                scale_grad_by_freq, mode,
                sparse, per_sample_weights,
                include_last_offset, padding_idx)

        # with Pool(num_chunks) as p:
        #     results = p.map(embedding_bag_chunk, zip(split_indices, split_offsets))

        # Execute in parallel using threads
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = [executor.submit(embedding_bag_chunk, (idx, off))
                       for idx, off in zip(split_indices, split_offsets)]
            results = [future.result() for future in futures]

        # Concatenate results to return final output
        return torch.cat([res[0] for res in results], dim=0), \
            torch.cat([res[1] for res in results], dim=0), \
            torch.cat([res[2] for res in results], dim=0), \
            torch.cat([res[3] for res in results], dim=0)
    return parallel_embedding_bag_horizontal()


def get_function_default_kwargs(func):
    """
    Extract default argument values for a function.
    Returns a dictionary mapping argument names to default values.
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


# plotted = {}


def profiled(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # new_gm = copy.deepcopy(gm)
    new_gm = gm

    def wrapped(fn, i):
        name = f"{fn.__name__}:{i}"
        # if name not in plotted:
        #     plotted[name] = tracy.plot_config(
        #         name, tracy.PlotFormatType.Number, True)

        def with_wrap(*args, **kwargs):
            with tracy.ScopedZone(name=name) as zone:
                # tracy.plot(plotted[name], 1)
                with profile(
                        activities=[ProfilerActivity.CPU,
                                    ProfilerActivity.CUDA],
                        profile_memory=True, record_shapes=True) as prof:
                    result = fn(*args, **kwargs)
                # tracy.plot(plotted[name], 0)
                total = prof.key_averages().total_average()
                zone.text(f"cpu-time: {total.self_cpu_time_total}")
                zone.text(f"cpu-mmry: {total.self_cpu_memory_usage}")
                zone.text(f"dev-time: {total.self_device_time_total}")
                zone.text(f"dev-mmry: {total.self_device_memory_usage}")
            return result
        return with_wrap

    for i, node in enumerate(new_gm.graph.nodes):
        if node.op == 'call_function':
            with new_gm.graph.inserting_before(node):
                new_node = new_gm.graph.call_function(
                    wrapped(node.target, i), args=node.args, kwargs=node.kwargs)
                node.replace_all_uses_with(new_node)
                new_gm.graph.erase_node(node)

    new_gm.graph.lint()
    new_gm.recompile()
    return new_gm


def parallelize(gm: torch.fx.GraphModule, topo=None, profile=None) -> torch.fx.GraphModule:
    # new_gm = copy.deepcopy(gm)
    new_gm = gm  # Directly modify the original gm

    # Get the default kwargs for parallel_embedding_bag
    default_kwargs = get_function_default_kwargs(parallel_embedding_bag)

    # Replace EmbeddingBag forward with parallelized embedding bag
    for node in new_gm.graph.nodes:
        if (node.op == 'call_function' and
                node.target == torch.ops.aten._embedding_bag.default):
            # Extract the args and kwargs from the original node
            args = node.args[:3]
            kwargs = dict(node.kwargs)

            # Merge the default kwargs of parallel_embedding_bag
            # Only use the defaults if they are not already in kwargs
            for key, default_value in default_kwargs.items():
                if key not in kwargs:
                    kwargs[key] = default_value

            # Insert the new node with parallel_embedding_bag
            with new_gm.graph.inserting_after(node):
                new_node = new_gm.graph.call_function(
                    parallel_embedding_bag,
                    args=args,
                    kwargs=kwargs
                )

                # Replace the original node's output with the new one
                node.replace_all_uses_with(new_node)

                # Remove the old EmbeddingBag node from the graph
                new_gm.graph.erase_node(node)

    # Finalize changes in the graph
    new_gm.graph.lint()  # Optional, checks for any graph invariants
    new_gm.recompile()  # Recompile the modified graph into the GraphModule

    return new_gm


def draw_graph(gm, save_path):
    g = graph_drawer.FxGraphDrawer(gm, 'training')
    with open(save_path, "w") as f:
        f.write(g.get_main_dot_graph().__str__())


def load_data(iteration):
    print(f"it {iteration}: data loaded")
    return iteration ** 2


def calc_loss(iteration):
    print(f"it {iteration}: loss calculated")


class PPRuntime:
    def __init__(
            self, model: Optional[GraphModule] = None,
            input_fn: Optional[Callable] = None,
            output_fn: Optional[Callable] = None,
            split_config: Optional[Dict[Any, int]] = None) -> None:
        """
        Args:
            input_fn: function exectued at each step before the pipeline runs,
                whose result will be fed into the model. Only stage 0 executes
                this function.
            output_fn: function executed at each sten after the model starts to
                produce results, whose value will be returned by `step`. Only
                the last stage executes this function.
            split_cfg: Map nodes to workers

        Note:
            input_fn and args_fn must be defined in a submodule instead of
            the main script. See https://stackoverflow.com/questions/41385708
        """
        mp.set_start_method("forkserver")

        self.split_config = copy.deepcopy(split_config)
        """How to split the compute graph"""

        self.num_stages = (
            1 if split_config is None else
            len(set(split_config.values())))
        """Numbero of stages == number of workers"""

        self.prereq_data = {
            i: mp.Manager().Queue() for i in range(1, self.num_stages)}
        """Queues that stores data of the previous stage"""

        self.step_sem = mp.Semaphore(0)
        """Semaphore signaling that .step() called. Used by the process 0"""

        self.input_fn = input_fn or (lambda x: print(f"Iter {x} Started"))
        self.output_fn = output_fn or (lambda x: print(f"Iter {x} Finished"))

        # Maybe start MP here since we need dynamic scaling
        self.processes = mp.spawn(
            self.worker_loop, args=(),
            nprocs=self.num_stages, join=False)

    def init_rank(self, rank):
        # TODO: split graph module
        pass

    def worker_loop(self, rank: int):
        """Each worker's own compute loop"""
        self.init_rank(rank)

        iteration = 0
        while True:

            # Obtain input
            if rank == 0:
                print(f"Step sem value is {self.step_sem.get_value()}")
                self.step_sem.acquire()
                data = self.input_fn(iteration)
            else:
                data = self.prereq_data[rank].get()

            # Process input
            with open(f"logs/{rank}.log", "a") as log:
                log.write(
                    f"rank {rank}: I'm {os.getpid()}, data is {data}\n")

            time.sleep(1 + random.random())
            data += 1

            # Send/process output
            if rank != self.num_stages - 1:
                self.prereq_data[rank+1].put(data)
            else:
                self.output_fn(iteration)

            iteration += 1

    # Advance all pipeline
    def step(self):
        """
        Step the pipeline, advance each stage in parallel.
        Must be called separately by each process to step its corresponding worker
        """
        self.step_sem.release()

    @staticmethod
    def profile_guided_split(
            model: GraphModule, data_loader: DataLoader, profile_result):
        """
        Generate split config given a profile_result
        Returns:
            a split_config
        """
        pass
