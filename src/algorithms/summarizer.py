
class OnlineSummarizer(object):
  def __init__(self, summary_length):
    self._summary_length = summary_length
    self._points = []
    self._summary = None
    self._current_mean = None
    self._size = 0

  def _output_all(self):
    self._summary = list(range(self._size))
    return self._summary

  def get_summary(self):
    return self._summary

  def update_summary(self, input_point):
    raise NotImplementedError("Abstract method")