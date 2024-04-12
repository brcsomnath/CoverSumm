import sys
sys.path.append("../")

import math
import heapq
import random
import numpy as np

from sklearn.metrics import pairwise_distances
from algorithms.summarizer import OnlineSummarizer

def l2_dist(X1,X2):
  """L2 distance."""
  return pairwise_distances(X1, X2, metric='l2', n_jobs=1)


class TopRankSummarizer(OnlineSummarizer):
  def __init__(self, 
              dist_func=l2_dist,
              summary_length=20,
              min_capacity=100):
    """
    Args:
      data: representation set
      dist_func   : Function to evaluate the distances      
    """
    super().__init__(summary_length=summary_length)
    self.dist_func = dist_func
    self.summary_length = summary_length
    self._min_capacity = min_capacity
    self.estimated_avg_distances = None
    self.num_samples = 0
    self.delta = 0.
    self.pivot = None
    self.l = 10

  def quick_select(self, arr, k, prior_pivot=None):
    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr) if prior_pivot is None else prior_pivot
    lows = [el for el in arr if el < pivot]
    highs = [el for el in arr if el > pivot]
    pivots = [el for el in arr if el == pivot]

    if k < len(lows):
      return self.quick_select(lows, k)
    elif k < len(lows) + len(pivots):
      return pivots[0]
    else:
      return self.quick_select(highs, k - len(lows) - len(pivots))
    
  def toprank(self):
    data = np.array(self._points)

    # Step 1: Use the RAND algorithm to obtain estimated average distances for each vertex
    sampled_pos =  np.array(np.random.choice(self._size, size=self.l), dtype='int')    
    S = data[sampled_pos]

    self.estimated_avg_distances = np.mean(
                                          self.dist_func(data, data[sampled_pos]),
                                          axis=1)
    self.num_samples += self.l
    
    # Step 3: Find vk and calculate Δ
    self.delta = 2 * min(np.max(self.dist_func(data, S), axis=0))

    f = math.sqrt(math.log2(self._size)/self.l)
    # threshold = sorted_vertices[vk][1] + 2 * f * delta
    dk = self.quick_select(self.estimated_avg_distances, 
                           self.summary_length-1, 
                           self.pivot)
    threshold = dk + 2 * f * self.delta
    self.pivot = dk
    
    # Step 4: Compute the candidate set E
    E = [v for v, avg_dist in enumerate(self.estimated_avg_distances) if avg_dist <= threshold]
    exact_avg_distances = np.mean(self.dist_func(data[E], data), axis=1)
    top_k_vertices = [E[idx] for idx in np.argsort(exact_avg_distances)[:self.summary_length]]
    return top_k_vertices

  def update_summary(self, input_point):
    self._points.append(input_point)
    self._size += 1
    
    if self._size <= self._summary_length:
      self._summary = self._output_all()
      return self._summary
    
    self._summary = self.toprank()
    return self._summary


def unit_test():
  data = np.random.uniform(-1, 1, size=(10000, 100))
