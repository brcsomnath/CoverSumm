import hnswlib
import numpy as np

from algorithms.summarizer import OnlineSummarizer

class HNSWOnlineSummarizer(OnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=summary_length)
    self._cover_tree = None

  def _return_knn(self, query):
    idx, dist = self._cover_tree.knn_query(
        query.reshape(1, -1), k=self._summary_length)
    return idx[0]
  
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
    
    self._size += 1

    if self._size <= self._summary_length + 1:
      self._points.append(input_point)

    if self._size <= self._summary_length:
      return self._output_all()

    self._update_mean(input_point)

    if self._cover_tree is None:
      self._cover_tree = hnswlib.Index(space='l2', dim=len(input_point))
      self._cover_tree.init_index(max_elements=12000, ef_construction=200, M=16)
      self._cover_tree.add_items(self._points)
    else:
      self._cover_tree.add_items(input_point[None, :])

    self._summary = self._return_knn(self._current_mean)
    return self._summary
