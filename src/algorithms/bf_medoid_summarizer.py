import numpy as np

from algorithms.summarizer import OnlineSummarizer
from scipy.spatial.distance import pdist, squareform


class BFMediodOnlineSummarizer(OnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=20)

  
  def find_medoids(self):
    """
    Finds the medoid of a set of high-dimensional points.

    Returns:
        int: The index of the medoid in the input array.
    """
    # Compute the pairwise distance matrix
    dist_matrix = squareform(pdist(self._points))

    # Sum the distances along each row (axis=1)
    total_distances = np.sum(dist_matrix, axis=1)

    # Find the index of the point with the smallest total distance
    medoid_candidates = np.argsort(total_distances)[:self._summary_length]

    return medoid_candidates
  
  
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

    self._summary = self.find_medoids()
    return self._summary
