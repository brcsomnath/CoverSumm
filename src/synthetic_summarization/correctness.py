import sys

sys.path.append("../")

import time
import pickle
import argparse
import collections
import faulthandler
faulthandler.enable()


import numpy as np
from tqdm import tqdm

from algorithms.bf_summarizer import BFOnlineSummarizer
from algorithms.naivesgt_summarizer import SGTreeOnlineSummarizer
from algorithms.fastsgt_summarizer import DecayCoverSummOnlineSummarizer
from algorithms.coversumm_summarizer import CoverSummOnlineSummarizer
from algorithms.lexrank_summarizer import LexRankOnlineSummarizer
from algorithms.lsa_summarizer import LSAOnlineSummarizer
from algorithms.sumbasic_summarizer import SumBasicOnlineSummarizer
from algorithms.centroid_opt_summarizer import CentroidOPTOnlineSummarizer
from algorithms.random_summarizer import RandomCoverSummOnlineSummarizer
from algorithms.dist_summarizer import DistCoverSummOnlineSummarizer
from algorithms.coversummworange_summarizer import CoverSummWORangeOnlineSummarizer
from algorithms.efficient_rangesgt_summarizer import LazyCoverSummOnlineSummarizer
from algorithms.meddit_summarizer import MedditSummarizer
from algorithms.toprank_summarizer import TopRankSummarizer
from algorithms.bf_medoid_summarizer import BFMediodOnlineSummarizer
from algorithms.hnsw_summarizer import HNSWOnlineSummarizer
from algorithms.faiss_summarizer import FaissOnlineSummarizer

def get_summarizer(name, dim=100):
  summarizer = None
  if name == 'bf':
    summarizer = BFOnlineSummarizer()
  elif name == 'naive_ct':
    summarizer = SGTreeOnlineSummarizer()
  elif name == 'decay':
    summarizer = DecayCoverSummOnlineSummarizer()
  elif name == 'coversumm':
    summarizer = CoverSummOnlineSummarizer(dim=dim)
  elif name == 'lexrank':
    summarizer = LexRankOnlineSummarizer()
  elif name == 'centroid_opt':
    summarizer = CentroidOPTOnlineSummarizer()
  elif name == 'lsa':
    summarizer = LSAOnlineSummarizer()
  elif name == 'sumbasic':
    summarizer = SumBasicOnlineSummarizer()
  elif name == 'random':
    summarizer = RandomCoverSummOnlineSummarizer()
  elif name == 'dist':
    summarizer = DistCoverSummOnlineSummarizer()
  elif name == 'lazy':
    summarizer = LazyCoverSummOnlineSummarizer(dim=dim)
  elif name == 'coversumm_wo_range':
    summarizer = CoverSummWORangeOnlineSummarizer()
  elif name == 'meddit':
    summarizer = MedditSummarizer()
  elif name == 'toprank':
    summarizer = TopRankSummarizer()
  elif name == 'bf_medoid':
    summarizer = BFMediodOnlineSummarizer()
  elif name == 'hnsw':
    summarizer = HNSWOnlineSummarizer()
  elif name == 'faiss':
    summarizer = FaissOnlineSummarizer()
  return summarizer


def load_pkl(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


def generate_synthetic_data(dim=100, N=100000):
  mean = np.zeros(dim)
  cov = np.eye(dim)

  points = np.random.uniform(low=-1, high=1, size=(N, dim))
  points = points.astype(dtype=np.float32)
  points = points / np.linalg.norm(points)
  return points

def generate_adv_data(dim=100, N=10000):
    mean = np.zeros(dim)
    cov = np.eye(dim)

    points = []
    for _ in range(N):
        mean = np.zeros(dim) + (100 * _/N) * np.ones(dim) 
        sample = np.random.multivariate_normal(mean, cov, size=1)
        points.append(sample[0])
    return np.array(points, dtype=np.float32)

def generate_lda_data(num_topics = 10,
                      vocab_size =100,
                      avg_doc_len = 150,
                      num_docs = 10000):

    alphas = [1] * num_topics
    betas = [1] * vocab_size


    phi = np.random.dirichlet(betas, size = num_topics)
    doc_lens = np.random.poisson(avg_doc_len, num_docs)

    representations = []
    for doc_id in tqdm(range(num_docs)):
        theta = np.random.dirichlet(alphas, 1)[0]

        review_rep = [0] * vocab_size
        for word_id in range(doc_lens[doc_id]):
            topic = np.argmax(np.random.multinomial(1, theta, 1)[0])
            word = np.random.multinomial(1, phi[topic], 1)
            review_rep += word[0]
        review_rep = review_rep / doc_lens[doc_id]
        representations.append(review_rep)
    return np.array(representations, dtype=np.float32)

def test_correctness(points, summarizer = CoverSummOnlineSummarizer()):
    bf_summarizer = BFOnlineSummarizer()

    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    correct = 0
    total = 0

    for i in tqdm(range(points.shape[0])):
        bf_summ = bf_summarizer.update_summary(points[i])
        summ = summarizer.update_summary(points[i])
        match = compare(summ, bf_summ)
        correct += match
        total += 1
    
    return correct / total

def test_correctness_medoid(points, summarizer = CoverSummOnlineSummarizer()):
    bf_summarizer = BFMediodOnlineSummarizer()

    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    correct = 0
    total = 0

    for i in tqdm(range(points.shape[0])):
        bf_summ = bf_summarizer.update_summary(points[i])
        summ = summarizer.update_summary(points[i])
        match = int(bf_summ[0] == summ[0])
        correct += match
        total += 1
    
    return correct / total

if __name__ == '__main__':
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarizer",
                        default="coversumm",
                        type=str,
                        help="Name of Summarizer.")
    parser.add_argument("--distr",
                        default="uniform",
                        type=str,
                        help="Distribution that vectors are sampled from.")
    parser.add_argument("--max_iter",
                        default=5,
                        type=int,
                        help="Number of runs.")
    parser.add_argument("--num_samples",
                        default=10000,
                        type=int,
                        help="Number of samples.")
    parser.add_argument("--mode",
                        default="centroid",
                        type=str,
                        help="Comparing w/ centroid or medoid.")

    args = parser.parse_args()
    
    # generate data
    if args.distr == 'uniform':
        points = generate_synthetic_data()
    elif args.distr == 'adv':
        points = generate_adv_data()
    elif args.distr == 'lda':
        points = generate_lda_data()
    
    points = np.unique(points, axis=0)

    summarizer = get_summarizer(args.summarizer, dim=points.shape[1])

    # run summarization
    if args.mode == 'centroid':
      percentage = test_correctness(points[:args.num_samples], 
                                    summarizer = summarizer)
    elif args.mode == 'medoid':
      percentage = test_correctness_medoid(points[:args.num_samples], 
                                           summarizer = summarizer)
    
    print(f"Correct outputs - {percentage}")
