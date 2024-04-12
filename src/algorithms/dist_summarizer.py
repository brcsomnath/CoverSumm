import math
import random
import numpy as np

from algorithms.random_summarizer import RandomCoverSummOnlineSummarizer

class DistCoverSummOnlineSummarizer(RandomCoverSummOnlineSummarizer):
  """
  Selects sentences in the local reservoir based on the distance 
  to the current mean. At each time step, updates the current mean 
  $\mu_t$ and returns k-nearest neighbours from the local reservoir. 
  """

  def __init__(self, capacity=100, threshold=5.5):
    super().__init__()
    self._current_neighbours = None
    self._current_neighbours_idx = None
    self._capacity = capacity
    self._threshold = threshold
  
  def _get_knn(self, query, size):
    """
    Given a query, returns kNN from local reservoir.

    Args:
      query: query vector for kNN search
    
    Returns:
      neighbours: k-nearest neighbours
      neighbour_idx: indices of the kNN output
    """
    distances = np.linalg.norm(self._current_neighbours - query, axis=-1)
    order = np.argsort(distances)[:size]

    neighbours = [self._current_neighbours[o] for o in order]
    neighbour_idx = [self._current_neighbours_idx[o] for o in order]
    self._threshold = max(distances)
    return neighbours, neighbour_idx

  def update_summary(self, input_point):
    self._size += 1

    if self._size <= self._summary_length + 1:
      self._points.append(input_point)

    if self._size <= self._summary_length:
      return self._output_all()

    self._update_mean(input_point)

    if self._current_neighbours is None:
      self._last_mean = self._current_mean
      self._current_neighbours = self._points
      self._current_neighbours_idx = list(range(len(self._points)))

      distances = np.linalg.norm(self._current_neighbours - self._current_mean, axis=-1)
      self._threshold = max(distances)
    
    # retain the k closest points to the current mean
    if len(self._current_neighbours_idx) >= self._capacity:
      self._current_neighbours, self._current_neighbours_idx = self._get_knn(
        self._current_mean, size=self._capacity)

    # accept points with distance within a threshold
    dist = np.linalg.norm(input_point - self._current_mean)
    if dist <= self._threshold:
      self._current_neighbours = np.append(self._current_neighbours, input_point[None, :], axis=0)
      self._current_neighbours_idx = np.append(self._current_neighbours_idx, self._size - 1)

    self._summary = self._get_summary(self._current_mean)
    return self._summary
