import math

import numpy as np
import graphgrove as gg

from copy import deepcopy
from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from algorithms.naivesgt_summarizer import SGTreeOnlineSummarizer


class LazyCoverSummOnlineSummarizer(SGTreeOnlineSummarizer):
  def __init__(self, dim=100, min_capacity=100, alpha = 1e-1):
    super().__init__()
    self._current_neighbours = None
    self._current_neighbours_idx = None
    self._dim = dim
    self._max_dist = 1e6
    self._threshold = 0.
    self._capacity = 100
    self._min_capacity = min_capacity
    self._alpha = alpha
    self._last_mean = None
    self._cache = []

  def _return_knn(self, query):
    idx, dist = self._cover_tree.kNearestNeighbours(query.reshape(1, -1),
                                                    k=self._summary_length)
    self._max_dist = max(dist[0])
    self._current_neighbours_idx = idx[0]
    return self._current_neighbours_idx

  def _range_query(self, query, radius):
    idx, dist, neighbours = self._cover_tree.RangeSearch(
        query.reshape(1, -1), r=radius, return_points=True)
    self._current_neighbours = neighbours[0]
    self._current_neighbours_idx = idx[0]
    self._max_dist = max(dist[0])

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
      self._cover_tree = SGTree_NNS_L2.from_matrix(np.array(
          self._points))
      self._last_mean = self._current_mean
      self._current_neighbours = deepcopy(self._points)
      self._current_neighbours_idx = list(range(self._size))

    if self._size <= self._min_capacity:
      if self._size > self._summary_length + 1:
        self._cover_tree.insert(input_point[None, :])
      self._summary = self._return_knn(self._current_mean)
      self._last_mean = self._current_mean
      return self._summary
    
    self._cache.append(input_point)
    if self._size > self._min_capacity:
        drift = np.linalg.norm(self._last_mean - self._current_mean)
        
        if drift >= self._threshold/2 or len(self._current_neighbours_idx) >= self._capacity:
          batch = np.array(self._cache, dtype=np.float32)
          self._cover_tree.insert(batch)
          self._cache = [] # empty cache

          delta = self._min_capacity / self._size
          self._threshold = math.sqrt(
              self._alpha * self._dim * math.log2(2 / delta) / 2 / self._size)
          
          self._summary = self._return_knn(self._current_mean)

          radius = self._threshold + self._max_dist

          self._range_query(self._current_mean, radius=radius)
          self._last_mean = self._current_mean
        else:
          if len(self._current_neighbours_idx) < self._capacity:
            inp_dist = np.linalg.norm(self._last_mean - input_point)
            if inp_dist < self._max_dist:
              self._current_neighbours = np.append(self._current_neighbours, input_point[None, :], axis=0)
              self._current_neighbours_idx = np.append(self._current_neighbours_idx, self._size - 1)
    
    self._summary = self._get_summary(self._current_mean)
    return self._summary
