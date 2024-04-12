import time
import json
import nltk
import torch
import argparse
import collections

import numpy as np

import sys
sys.path.append("../../")

from utils.data import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.summary import truncate_summary, RougeEvaluator

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


def load_json(filename):
  with open(filename) as file:
    return json.load(file)


def load_representations(data, product_id):
  representations = []
  
  for review in tqdm(data[product_id]):
    sentences = nltk.sent_tokenize(review['review_body'])
    if not sentences:
      continue
    
    batch = tokenizer(sentences,
                    pad_to_max_length=True,
                    truncation=True,
                    add_special_tokens=True)
    input_ids = torch.LongTensor(batch['input_ids']).to(device)
    attention_mask = torch.LongTensor(batch['attention_mask']).to(device)
    output = model(input_ids, attention_mask=attention_mask)
    output = output['pooler_output'].detach().cpu().numpy()
    representations.append(output)
  return representations


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
  return summarizer


def online_summary(points, summarizer=BFOnlineSummarizer()):
  for i in range(points.shape[0]):
    summ = summarizer.update_summary(points[i])
  return summ


def run_online_summarization(points, summarizer=BFOnlineSummarizer()):
  import time # adhoc fix. TODO:find the root cause of this bug
  start = time.time()
  for i in (range(points.shape[0])):
    summ = summarizer.update_summary(points[i])
  return time.time() - start


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--summarizer",
                      default="coversumm",
                      type=str,
                      help="Name of Summarizer.")
  parser.add_argument("--model_name",
                      default="bert-base-uncased",
                      type=str,
                      help="Bert model name.")
  parser.add_argument("--data_path",
                      default="../../../data/amazon/amazon_us_reviews.json",
                      type=str,
                      help="Path to dataset.")

  args = parser.parse_args()

  device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

  tokenizer = BertTokenizer.from_pretrained(args.model_name)
  model = BertModel.from_pretrained(args.model_name)
  model.to(device)


  data = load_json(args.data_path)

  total_time = 0
  count = 0
  for product_id in list(data.keys()):
    count += 1
    representations = load_representations(data, product_id)

    summarizer = get_summarizer(args.summarizer)

    for review_rep in tqdm(representations):
      points = review_rep.astype(np.float32)
      runtime = run_online_summarization(points, summarizer)
      total_time += runtime
    del representations
    del summarizer
  
  print(f"Amortized runtime: {total_time / count}")