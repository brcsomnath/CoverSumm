import math
import random
import numpy as np

from copy import deepcopy
from algorithms.bf_summarizer import BFOnlineSummarizer

class RandomCoverSummOnlineSummarizer(BFOnlineSummarizer):
    """
    Randomly selects sentences to be in the local reservoir.
    At each time step, updates the current mean $\mu_t$ and returns
    k-nearest neighbours from the local reservoir. 
    """

    def __init__(self, capacity=100):
        super().__init__()
        self._current_neighbours = None
        self._current_neighbours_idx = None
        self._capacity = capacity

    def _get_summary(self, query):
        """
        Given a query, returns kNN from local reservoir.

        Args:
            query: query vector for kNN search
        
        Returns:
            neighbour_idx: indices of the kNN output
        """
        distances = np.linalg.norm(self._current_neighbours - query, axis=-1)
        order = np.argsort(distances)[:self._summary_length]
        neighbour_idx = [self._current_neighbours_idx[o] for o in order]
        return neighbour_idx

    def update_summary(self, input_point):
        self._size += 1

        if self._size <= self._summary_length + 1:
            self._points.append(input_point)

        if self._size <= self._summary_length:
            return self._output_all()

        self._update_mean(input_point)

        if self._current_neighbours is None:
            self._last_mean = self._current_mean
            self._current_neighbours = deepcopy(self._points)
            self._current_neighbours_idx = list(range(self._size))


        if len(self._current_neighbours) >= self._capacity:
          self._current_neighbours = self._current_neighbours[1:]
          self._current_neighbours_idx = self._current_neighbours_idx[1:]
        
        # selects a sentence to be part of the reservoir randomly
        if random.random() > 0.5:
            self._current_neighbours = np.append(self._current_neighbours, input_point[None, :], axis=0)
            self._current_neighbours_idx = np.append(self._current_neighbours_idx, self._size - 1)
    
        self._summary = self._get_summary(self._current_mean)
        return self._summary
