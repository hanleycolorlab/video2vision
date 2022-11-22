from collections import Counter
from contextlib import suppress
import json
import time
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np

from .io import Loader, OutOfInputs, _read_write_from_to_buffer, Writer
from .operators import Operator, OPERATOR_REGISTRY

__all__ = ['load_pipeline', 'Pipeline']


class Pipeline(nx.DiGraph):
    '''
    :class:`Pipeline` instantiates a full image processing pipeline. It
    subclasses :class:`networkx.DiGraph`, which represents a directed graph.
    Each node in the graph represents an image processing :class:`Operator`.
    Each directed edge in the graph represents the output of one operator being
    passed to another operator as input.

    Since networkx may not be familiar to everyone, we provide detailed notes
    on the parts of :class:`networkx.DiGraph` we are using. First: each node in
    the graph must be represented by a hashable object. We use integer indices.
    The nodes property returns an iterator over all of the nodes. In our case,
    that means it iterates over the integer indices, not the underlying
    :class:`Operator` s. In addition, a dictionary is associated to each node,
    which is where we store the actual :class:`Operator`. This dictionary can
    be accessed by:

    .. code-block:: python

        pipeline.nodes[operator_idx]

    The :class:`Operator` is stored under the key 'operator':

    .. code-block:: python

        pipeline.nodes[operator_idx]['operator']

    We also store inputs to the operator under the key 'inputs':

    .. code-block:: python

        pipeline.nodes[operator_idx]['inputs']

    This key will only be generated when we start running the pipeline. Its
    value will be a list of length equal to the number of expected inputs to
    the :class:`Operator`, and will initially be filled with Nones. The outputs
    of earlier nodes will be placed in that list as they are generated, to
    store them until it is time to run that :class:`Operator`.

    Nodes can be added to the graph using the :meth:`networkx.DiGraph.add_node`
    method. The contents of the dictionary can also be appended as keyword
    arguments:

    .. code-block:: python

        op = Operator(...)
        pipeline.add_node(operator_idx, operator=op)

    Edges are stored as pairs, where the first entry in the pair is the node it
    emerges from and the second entry is the node it enters into. Again, in our
    case, those are the indices, not the :class:`Operator` s. The edges also
    have dictionaries associated to them, which can be accessed by:

    .. code-block:: python

        pipeline.edges[from_idx, to_idx]

    Since the inputs to an :class:`Operator` must be ordered, we keep track of
    which input a particular output is going to in that dictionary:

    .. code-block:: python

        pipeline.edges[from_idx, to_idx]['in_slot']

    Edges can be added similarly to nodes:

    .. code-block:: python

        pipeline.add_edge(from_idx, to_idx, in_slot=in_slot)
    '''
    # TODO: Add more error checking. A lot more.

    def __call__(self, *images) -> Union[List[np.ndarray], np.ndarray]:
        # Put images into buffers
        for loader, _images in zip(self.get_loaders(), images):
            loader.buff = [_images]

        # Run pipeline
        with _read_write_from_to_buffer():
            self.run()

        # Retrieve output
        out = [writer.buff.pop() for writer in self.get_writers()]

        return out[0] if (len(out) == 1) else out

    def add_operator(self, operator: Operator, idx: Optional[int] = None) \
            -> int:
        '''
        This adds an :class:`Operator` to the pipeline, and returns the
        operator's index.
        '''
        if idx is None:
            idx = max(self.nodes, default=-1) + 1
        if idx in self.nodes and 'operator' in self.nodes[idx]:
            raise ValueError(f'Duplicate indices: {idx}')
        self.add_node(idx, operator=operator)
        return idx

    @classmethod
    def chain(cls, *operators):
        '''
        This is a convenience function for building a :class:`Pipeline` where
        you have a list of :class:`Operator` s to be applied in sequence,
        without other changes.
        '''
        # First, instantiate the graph with a list of edges.
        edges = [(i, i + 1, {'in_slot': 0}) for i in range(len(operators) - 1)]
        pipe = cls(edges)
        # And add the operators.
        for node_idx, operator in enumerate(operators):
            pipe.nodes[node_idx]['operator'] = operator
        return pipe

    def get_loaders(self) -> List[Loader]:
        '''
        Returns a list of the :class:`Loader` s in the :class:`Pipeline`,
        ordered by index.
        '''
        idxs = [i for i in self.nodes
                if isinstance(self.nodes[i]['operator'], Loader)]
        return [self.nodes[i]['operator'] for i in sorted(idxs)]

    def get_writers(self) -> List[Writer]:
        '''
        Returns a list of the :class:`Writer` s in the :class:`Pipeline`,
        ordered by index.
        '''
        idxs = [i for i in self.nodes
                if isinstance(self.nodes[i]['operator'], Writer)]
        return [self.nodes[i]['operator'] for i in sorted(idxs)]

    @classmethod
    def load(cls, path: str):
        '''
        Loads a serialized :class:`Pipeline` from disk.

        Args:
            path (str): Path to serialized :class:`Pipeline`.
        '''
        with open(path, 'r') as in_file:
            graph = json.load(in_file)

        pipe = cls()

        # TODO: Is there a more efficient way to do this?

        for node in graph:
            op_cls = OPERATOR_REGISTRY.get(node['operator'].pop('class'))
            op = op_cls(**node['operator'])
            pipe.add_operator(op, idx=node['index'])
            for to_idx, details in node['edges']:
                pipe.add_edge(node['index'], to_idx, **details)

        return pipe

    def release_writers(self):
        '''
        Releases the writers. This should be called when writing video to disk,
        after you have written all the frames you need to write. Without this
        call, the video will not save properly. This method is called by the
        :meth:`run` method automatically when it's finished.
        '''
        for writer in self.get_writers():
            writer.release()

    def run(self) -> Dict:
        '''
        This runs the :class:`Pipeline` until it runs out of input images. It
        returns a dictionary of runtimes.
        '''
        start_time = time.perf_counter()
        op_times = Counter()

        # Add empty input slots
        for node in self.nodes.values():
            node['inputs'] = [None] * node['operator'].num_inputs

        # The topological sort returns an iterator over the node
        # indices ordered so that we will not reach any node until we
        # have already visited the nodes generating its inputs.
        op_idxs = list(nx.topological_sort(self))

        # TODO: Add check on matching lengths of Loaders

        # The Loaders will generate an OutOfInputs exception when they run out
        # of inputs. This gives us a convenient way to break out of the loops.
        with suppress(OutOfInputs):
            while True:
                # Go through the operators.
                for op_idx in op_idxs:
                    # Retrieve the dictionary corresponding to the operator
                    # index.
                    op: Dict = self.nodes[op_idx]
                    # Check the inputs have been filled in.
                    if any(x is None for x in op['inputs']):
                        raise RuntimeError(
                            f'Input missing from operator {op_idx}: {repr(op)}'
                            f'; pipeline graph must be incomplete.'
                        )
                    op_name = op['operator'].__class__.__name__
                    op_start_time = time.perf_counter()
                    # Run the operator on the inputs.
                    out = op['operator'](*op.pop('inputs'))
                    op_times[op_name] += (time.perf_counter() - op_start_time)

                    # Place the output into the corresponding input slots of
                    # operators downstream of this operator.
                    for _, down_op_idx in self.out_edges(op_idx):
                        in_slot = self.edges[op_idx, down_op_idx]['in_slot']
                        self.nodes[down_op_idx]['inputs'][in_slot] = out
                    # Reset the inputs to this operator.
                    op['inputs'] = [None] * op['operator'].num_inputs

        self.release_writers()

        total_runtime = time.perf_counter() - start_time
        op_times['pipeline'] = total_runtime - sum(op_times.values())

        return op_times

    def save(self, path: str):
        '''
        Serializes a :class:`Pipeline` to disk as a JSON file.

        Args:
            path (str): Path to serialize the :class:`Pipeline` to.
        '''
        out = []
        for idx in self.nodes:
            out.append({
                'index': idx,
                'operator': self.nodes[idx]['operator']._to_json(),
                'edges': list(self.succ[idx].items())
            })

        with open(path, 'w') as out_file:
            json.dump(out, out_file, indent=4)

    def set_all_paths(self, *paths):
        '''
        Sets all I/O paths.

        Args:
            paths: Each path should be a path to read from or write to, with
            the input paths preceding the output paths.
        '''
        loaders, writers = self.get_loaders(), self.get_writers()
        if len(paths) != len(loaders) + len(writers):
            raise ValueError(
                f'Incorrect number of paths: is {len(paths)}, should be '
                f'{len(loaders) + len(writers)}'
            )

        n_loaders = len(loaders)

        self.set_loader_paths(*paths[:n_loaders])
        self.set_writer_paths(*paths[n_loaders:])

    def set_batch_size(self, batch_size: int):
        '''
        Sets the batch size of the :class:`Loader` s.

        Args:
            batch_size (int): Number of frames to load per iteration.
        '''
        for loader in self.get_loaders():
            loader.batch_size = batch_size

    def set_loader_paths(self, *paths):
        '''
        Sets the paths for the loaders.
        '''
        loaders = self.get_loaders()
        if len(paths) != len(loaders):
            raise ValueError(
                f'Incorrect number of paths: is {len(paths)}, should be '
                f'{len(loaders)}'
            )

        for path, loader in zip(paths, loaders):
            loader.set_path(path)

    def set_writer_paths(self, *paths):
        '''
        Sets the paths for the writers.
        '''
        writers = self.get_writers()
        if len(paths) != len(writers):
            raise ValueError(
                f'Incorrect number of paths: is {len(paths)}, should be '
                f'{len(writers)}'
            )

        for path, writer in zip(paths, writers):
            writer.set_path(path)


load_pipeline = Pipeline.load
