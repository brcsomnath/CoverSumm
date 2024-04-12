from summa.summarizer import summarize
from algorithms.summarizer import OnlineSummarizer

# TextRank implementation using the package: https://github.com/summanlp/textrank

class TextRankOnlineSummarizer(OnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=summary_length)
    self._all_texts = []

  def update_summary(self, input_text):
    """Returns the updated summary once new text comes in.

    Args:
        input_text - incoming review text
    
    Returns:
        summary - updated summary of the points.
    """

    self._all_texts.append(input_text)
    self._size += 1

    text = " ".join(self._all_texts)

    if self._size <= self._summary_length:
      return text


    # textrank based summarizer
    self._summary = summarize(text, words=400)
    return self._summary
