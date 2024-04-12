import os
import sys
sys.path.append("../")

import collections
import pickle, time
import numpy as np
import pandas as pd
import matplotlib, datetime
import matplotlib.pyplot as plt
import scipy.sparse as sp_sparse

from tqdm import tqdm
from scipy.sparse import hstack
from sklearn.metrics import pairwise_distances
from algorithms.summarizer import OnlineSummarizer


def l2_dist(X1,X2):
  """L2 distance."""
  return pairwise_distances(X1, X2, metric='l2', n_jobs=1)


def l1_dist(X1,X2):
  """L1 distance."""
  return pairwise_distances(X1, X2, metric='l1', n_jobs=1)


def cosine_dist(X1, X2):
  """Cosine distance."""
  return pairwise_distances(X1, X2, metric='cosine', n_jobs=1)


class MedditSummarizer(OnlineSummarizer):

  def __init__(self, 
                dist_func=l2_dist,
                summary_length=20,
                min_capacity=50):
    """
    Args:
      data: representation set
      dist_func   : Function to evaluate the distances      
    """
    super().__init__(summary_length=summary_length)
    
    
    np.random.seed(0) #Random seed for reproducibility
    self.n         = 0
    self.num_init_pulls = 1
    self.Delta     = 1.0 
    self.num_arms  = 32 #Number of arms to be pulled in every round parallelly
    self.step_size = 32 #Number of distance evaluation to be performed on every arm
    self.dist_func = dist_func
    self.summary_length = summary_length
    self._min_capacity = min_capacity
    self.sigma = 1.0 
  
  def initialise(self, init_size=1):
    data = np.array(self._points)
    tmp_pos = np.array(np.random.choice(self.n, 
                                        size=init_size, 
                                        replace=False), 
                                        dtype='int')
    self.estimate = np.mean(self.dist_func(data, data[tmp_pos]), axis = 1)


  def _update_mean(self, input_point):
    """Updates the aggreates mean as new points come in.

    Args:
        input_point - new point inserted
    """
    N = self.n
    self._current_mean = np.mean(
        self._points, axis=0) if self._current_mean is None else (
            self._current_mean * (N-1) + input_point) / N


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

  def choose_arm(self):
    """
    Chooses the "num_arms" arms with lowest lcb and removes the ones which have been pulled n times.
    Returns None at stopping time
    """
    data = np.array(self._points)
    low_lcb_arms = np.argpartition(self.lcb, self.num_arms)[:self.num_arms]
        
    #Arms which are pulled >= ntimes and ucb!=lcb
    arms_pulled_morethan_n = low_lcb_arms[ np.where( 
      (self.T[low_lcb_arms] >= self.n) & (self.ucb[low_lcb_arms] != self.lcb[low_lcb_arms]) ) ]
    
    if arms_pulled_morethan_n.shape[0]>0:
      # Compute the distance of these arms accurately
      self.estimate[arms_pulled_morethan_n] = np.mean(self.dist_func(
                                                      data[arms_pulled_morethan_n], 
                                                      data), 
                                                      axis=1 )
      self.T[arms_pulled_morethan_n]  += self.n
      self.ucb[arms_pulled_morethan_n] = self.estimate[arms_pulled_morethan_n]
      self.lcb[arms_pulled_morethan_n] = self.estimate[arms_pulled_morethan_n]
    
    if self.ucb.min() <  self.lcb[np.argpartition(self.lcb,1)[1]]: #Exit condition
      return None

    arms_to_pull = low_lcb_arms[ np.where(self.T[low_lcb_arms] < self.n) ]
    return arms_to_pull


  def pull_arm(self, arms):
    """
    Pulls the "num_arms" arms "step_size" times. Updates the estimate, ucb, lcb
    """
    data = np.array(self._points)
    tmp_pos      = np.array( np.random.choice(
      self.n, size=self.step_size, replace=False), dtype='int')    
    X_arm        = data[arms]
    X_other_arms = data[tmp_pos]

    Tmean = np.mean(self.dist_func(X_arm, X_other_arms), axis=1)
    self.estimate[arms]   = (self.estimate[arms] * self.T[arms] + \
                             Tmean * self.step_size)/( self.T[arms] + self.step_size + 0.0)
    self.T[arms]          = self.T[arms] + self.step_size
    self.lcb[arms]        = self.estimate[arms] - np.sqrt(
      self.sigma ** 2 * np.log(1/self.Delta)/(self.T[arms]+0.0))
    self.ucb[arms]        = self.estimate[arms] + np.sqrt(
      self.sigma ** 2 * np.log(1/self.Delta)/(self.T[arms]+0.0))

  
  def get_medoids(self):
    self.initialise()

    threshold = np.sqrt(self.sigma**2 *np.log(1/self.Delta)/self.num_init_pulls)
    self.lcb = self.estimate - threshold
    self.ucb = self.estimate + threshold
    self.T = self.num_init_pulls * np.ones(self.n, dtype='int')

    #Step 2: Iterate
    num_iters = self.n * 10
    for ind in (range(num_iters)):   
      #Choose the arms
      arms_to_pull = self.choose_arm()
      
      #Stop if we have found the best arm
      if arms_to_pull is None:
        return [np.argmin(self.lcb)] # best arm
      
      if len(arms_to_pull) == 0:
        continue
      
      #Pull the arms
      self.pull_arm(arms_to_pull)
    return [np.argmin(self.lcb)]

  def update_summary(self, input_point):
    self.n += 1
    self._size = self.n
    self.Delta = 1/self.n
    self._points.append(input_point)
    
    self._update_mean(input_point)
    # if self.n <= self._summary_length:
    #   return self._output_all()

    if self.n < self._summary_length*2:
      self._summary = self._return_knn(self._current_mean)
      return self._summary
    
    self._summary = self.get_medoids()
    return self._summary