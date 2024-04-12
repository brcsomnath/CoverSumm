from algorithms.summarizer import OnlineSummarizer

import numpy as np

class BFOnlineSummarizer(OnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=20)
    self._cover_tree = None
  
  def _return_knn(self, query):
    """Returns the k-nearest neighbours given a query
    based on L2-distance between query and elements.
    
    Args:
        query - query for the kNN search
    
    Returns:
        knn_indices - indices of the k-nearest neighbours
    """

    dist = np.linalg.norm(self._points - query, axis=-1)
    knn_indices = np.argsort(dist)[:self._summary_length]
    return knn_indices

  def _update_mean(self, input_point):
    """Updates the aggreates mean as new points come in.

    Args:
        input_point - new point inserted
    """

    N = self._size
    self._current_mean = np.mean(
        self._points, axis=0) if self._current_mean is None else (
            self._current_mean * (N-1) + input_point) / N
  
  def update_summary(self, input_point):
    """Returns the updated summary once new text comes in.

    Args:
        input_point - incoming new point
    
    Returns:
        summary - updated summary of the points.
    """

    self._points.append(input_point)
    self._size += 1

    if self._size <= self._summary_length:
      return self._output_all()

    self._update_mean(input_point)

    self._summary = self._return_knn(self._current_mean)
    return self._summary
