import math
import time

import numpy as np
import graphgrove as gg

from copy import deepcopy
from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from algorithms.coversumm_summarizer import CoverSummOnlineSummarizer


class CoverSummWORangeOnlineSummarizer(CoverSummOnlineSummarizer):
  """CoverSumm if range query was not performed."""
  
  def __init__(self, dim=100, min_capacity=100, alpha = 1e-1, summary_length=20):
    super().__init__(summary_length=summary_length)
    self._current_neighbours = None
    self._current_neighbours_idx = None
    self._dim = dim
    self._max_dist = 1e6
    self._threshold = 0.
    self._capacity = 100
    self._min_capacity = min_capacity
    self._alpha = alpha
    self._last_mean = None

  def _return_knn(self, query):
    idx, dist, neighbours = self._cover_tree.kNearestNeighbours(query.reshape(1, -1),
                                                    k=self._summary_length,
                                                    return_points=True)
    self._max_dist = max(dist[0])
    self._current_neighbours = neighbours[0]
    self._current_neighbours_idx = idx[0]
    return self._current_neighbours_idx

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
    else:
      self._cover_tree.insert(input_point[None, :])

    if self._size <= self._min_capacity:
      self._summary = self._return_knn(self._current_mean)
      self._last_mean = self._current_mean
      return self._summary
        
    if self._size > self._min_capacity:
      drift = np.linalg.norm(self._last_mean - self._current_mean)
      
      if drift >= self._threshold/2 or len(self._current_neighbours_idx) >= self._capacity:
        delta = self._min_capacity / self._size
        self._threshold = math.sqrt(
            self._alpha * self._dim * math.log2(2 / delta) / 2 / self._size)
        
        self._summary = self._return_knn(self._current_mean)
        self._last_mean = self._current_mean
      else:
        if len(self._current_neighbours_idx) < self._capacity:
          inp_dist = np.linalg.norm(self._last_mean - input_point)
          if inp_dist < self._max_dist:
            self._current_neighbours = np.append(self._current_neighbours, input_point[None, :], axis=0)
            self._current_neighbours_idx = np.append(self._current_neighbours_idx, self._size - 1)
    
    self._summary = self._get_summary(self._current_mean)
    return self._summary
