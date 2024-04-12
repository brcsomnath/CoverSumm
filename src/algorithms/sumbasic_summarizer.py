from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.sum_basic import SumBasicSummarizer

from algorithms.summarizer import OnlineSummarizer

# SumBasic implementation using the sumy package

class SumBasicOnlineSummarizer(OnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=summary_length)
    self._summarizer = SumBasicSummarizer()
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
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # textrank based summarizer
    output = self._summarizer(parser.document, self._summary_length)
    self._summary = [str(sentence._text) for sentence in output]
    return self._summary
