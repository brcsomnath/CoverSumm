# +
import graphgrove as gg

from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from algorithms.naivesgt_summarizer import SGTreeOnlineSummarizer
# -

import collections
import queue
import math

import numpy as np

class DecayCoverSummOnlineSummarizer(SGTreeOnlineSummarizer):
    def __init__(self, min_capacity=100):
        super().__init__()
        self._min_level = -1
        self._is_updated = False
        self._last_mean = None 
        self._min_capacity = min_capacity
        self._current_neighbours = None
        self._current_neighbours_idx = None
    
    def _return_knn(self, query):
        idx, dist, neighbours = self._cover_tree.kNearestNeighbours(query.reshape(1, -1),
                                                        k=self._summary_length,
                                                        return_points=True)

        self._current_neighbours = neighbours[0]
        self._current_neighbours_idx = idx[0]
        return self._current_neighbours_idx
    

    def _get_summary(self, query):
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

        if self._cover_tree is None:
            self._cover_tree = SGTree_NNS_L2.from_matrix(np.array(self._points))
        else:
            self._cover_tree.insert(input_point[None, :])

        if self._size <= self._min_capacity:
            self._summary = self._return_knn(self._current_mean)
            self._last_mean = self._current_mean
        

        if self._size > self._min_capacity:
            drift = np.linalg.norm(self._last_mean - self._current_mean)

            cutoff_dist = 0.003 * np.exp(-1 * self._size/10000)

            if drift > cutoff_dist:
                self._summary = self._return_knn(self._current_mean)
                self._last_mean = self._current_mean
        
        self._summary = self._get_summary(self._current_mean)
        return self._summary
