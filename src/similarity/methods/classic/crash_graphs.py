from abc import abstractmethod
from typing import List, Tuple

from src.similarity.preprocess.seq_coder import SeqCoder


class Graph:
    @abstractmethod
    def add_edge(self, new_edge: Tuple[int, int]):
        raise NotImplementedError

    def build(self, vertices):
        start, end = None, None
        for vertex in vertices:
            if start is None:
                start = vertex
            elif end is None:
                end = vertex
            else:
                self.add_edge((start, end))
                start, end = end, vertex


class EdgeList(Graph):
    def __init__(self):
        self.edges = set()

    def add_edge(self, new_edge: Tuple[int, int]):
        self.edges.add(new_edge)


class CrashGraphs:
    def __init__(self, coder: SeqCoder):
        self.coder = coder
        self.ns = (1,)

    def fit(self, sim_train_data: List[Tuple[int, int, int]] = None, stack_ids: List[int] = None):
        self.coder.fit(stack_ids)

    def predict(self, anchor_id: int, stack_ids: List[int]) -> List[float]:
        group = EdgeList()
        for stack_id in stack_ids:
            group.build(self.coder.ngrams(stack_id, ns=self.ns).keys())

        stack_trace = EdgeList()
        stack_trace.build(self.coder.ngrams(anchor_id, ns=self.ns).keys())

        common_edges_num = len(group.edges.intersection(stack_trace.edges))
        score = common_edges_num / min(len(group.edges) + 1, len(stack_trace.edges) + 1)

        return [score] * len(stack_ids)
