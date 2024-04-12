import numpy as np

from scipy import spatial
from algorithms.summarizer import OnlineSummarizer
from sklearn.metrics.pairwise import cosine_similarity


# parts of the code has been borrowed from https://github.com/simpetre/lexrank

class LexRankOnlineSummarizer(OnlineSummarizer):
  def __init__(self, summary_length=20):
    super().__init__(summary_length=summary_length)
    self.cached_similarity_matrix = None
    self.cached_degrees = None

  def calc_power_method(self,
                        similarity_matrix,
                        degrees,
                        stopping_criterion=0.0005,
                        max_loops=1000):
      """Return PageRank scores of similarity matrix
      
      Args: 
          similarity_matrix: a matrix of calculated document similarities
          stopping_criterion: stops when the incremental difference in norms is less than this
          max_loops: sets the maximum number of times for the loop to run
      """
      p_initial = np.ones(shape=len(degrees))/len(degrees)
      i = 0
      # loop until no change between successive matrix iterations
      while True:
          i += 1
          p_update = np.matmul(similarity_matrix.T, p_initial)
          delta = np.linalg.norm(p_update - p_initial)
          if delta < stopping_criterion or i >= max_loops:
              break
          else:
              p_initial = p_update
      p_update = p_update / np.max(p_update)
      return p_update

  def calculate_similarity_mat(self, representations, similarity_threshold=0.05):
    """Returns a matrix representation of a graph of document similarities.

    Args:
      representations: text representations of the corpus
      similarity_threshold: values higher than this get a vertex in the graph
    
    Returns:
      cosine_similarities: cosine similarity matrix
      degrees: degree of the graph
    """

    representations = np.array(representations, dtype=np.float32)
    
    num_sents = representations.shape[0]
    cosine_similarities = np.empty(shape=(num_sents, num_sents))
    degrees = np.zeros(shape=(representations.shape[0]))

    if self.cached_similarity_matrix is not None and self.cached_degrees is not None:
      cosine_similarities[:num_sents-1, :num_sents-1] = self.cached_similarity_matrix

    # get cosine similarities between sentences
    for i, sent1 in enumerate(representations):
      for j, sent2 in enumerate(representations):
        if i < (num_sents - 1) and j < (num_sents - 1):
          continue

        calculated_cosine_similarity = 1 - spatial.distance.cosine(sent1, sent2)
        if calculated_cosine_similarity > similarity_threshold:
          cosine_similarities[i, j] = calculated_cosine_similarity
          degrees[i] += 1
        else:
          cosine_similarities[i, j] = 0

    cosine_similarities = cosine_similarities/degrees

    self.cached_similarity_matrix = cosine_similarities
    self.cached_degrees = degrees
    return cosine_similarities, degrees

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

    similarity_matrix, degrees = self.calculate_similarity_mat(self._points)
    lexrank_vector = self.calc_power_method(similarity_matrix, degrees)

    self._summary = np.argsort(lexrank_vector)[::-1][:self._summary_length]
    return self._summary
