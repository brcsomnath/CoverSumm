import numpy as np

from algorithms.bf_summarizer import BFOnlineSummarizer

# Implementation follows the paper: https://aclanthology.org/W17-4511.pdf

class CentroidOPTOnlineSummarizer(BFOnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=summary_length)

  
  def get_summary(self, centroid):
    summary = []
    for i in range(self._summary_length):
      if len(summary) > 0:
        summ_representations = np.sum([self._points[i] for i in summary])
      elif len(summary) == 0:
        summ_representations = np.zeros(len(self._points[0]))
      
      # mean summary representation
      updated_summ_rep = np.array(
        [(summ_representations + point) for point in self._points]) / (len(summary) + 1)
      dist = np.linalg.norm(updated_summ_rep - centroid, axis=-1)

      # select the best sentence not in the current summary
      for sent in np.argsort(dist):
        if sent not in summary:
          best_sent = sent
      summary.append(best_sent)
    return summary

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

    self._summary = self.get_summary(self._current_mean)
    return self._summary
