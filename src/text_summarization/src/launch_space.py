# +
import os
import time
import json
import torch
import argparse
import collections

import numpy as np

# +
import sys
sys.path.append("../../")

from utils.data import *
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.summary import truncate_summary, RougeEvaluator
# -

from tqdm import tqdm
from algorithms.bf_summarizer import BFOnlineSummarizer
from algorithms.naivesgt_summarizer import SGTreeOnlineSummarizer
from algorithms.fastsgt_summarizer import DecayCoverSummOnlineSummarizer
from algorithms.coversumm_summarizer import CoverSummOnlineSummarizer
from algorithms.lexrank_summarizer import LexRankOnlineSummarizer
from algorithms.lsa_summarizer import LSAOnlineSummarizer
from algorithms.meddit_summarizer import MedditSummarizer
from algorithms.toprank_summarizer import TopRankSummarizer
from algorithms.sumbasic_summarizer import SumBasicOnlineSummarizer
from algorithms.bf_medoid_summarizer import BFMediodOnlineSummarizer
from algorithms.dist_summarizer import DistCoverSummOnlineSummarizer
from algorithms.random_summarizer import RandomCoverSummOnlineSummarizer
from algorithms.centroid_opt_summarizer import CentroidOPTOnlineSummarizer
from algorithms.coversummworange_summarizer import CoverSummWORangeOnlineSummarizer
from algorithms.efficient_rangesgt_summarizer import LazyCoverSummOnlineSummarizer
from algorithms.hnsw_summarizer import HNSWOnlineSummarizer
from algorithms.faiss_summarizer import FaissOnlineSummarizer


def load_dataset(summ_data_path, spm_path):
  f = open(summ_data_path, 'r')
  summ_data = json.load(f)
  f.close()
  
  # prepare summarization dataset
  summ_dataset = ReviewSummarizationDataset(summ_data,
                                            spmodel=spm_path,
                                            max_rev_len=150,
                                            max_sen_len=40)
  vocab_size = summ_dataset.vocab_size
  pad_id = summ_dataset.pad_id()
  bos_id = summ_dataset.bos_id()
  eos_id = summ_dataset.eos_id()
  unk_id = summ_dataset.unk_id()

  # wrapper for collate function
  collator = ReviewCollator(padding_idx=pad_id,
                            unk_idx=unk_id,
                            bos_idx=bos_id,
                            eos_idx=eos_id)

  # split dev/test entities
  summ_dataset.entity_split(split_by='alphanum')

  # create entity data loaders
  summ_dls = {}
  summ_samplers = summ_dataset.get_entity_batch_samplers(args.batch_size)
  for entity_id, entity_sampler in summ_samplers.items():
    summ_dls[entity_id] = DataLoader(
        summ_dataset,
        batch_sampler=entity_sampler,
        collate_fn=collator.collate_reviews_with_ids)
  data = get_data(summ_dls, summ_dataset)
  return data

def np_encoder(object):
  if isinstance(object, np.generic):
    return object.item()

def dump_data(data, path='../data/space/sample_space_data.json'):
  directory = os.path.dirname(path)
  if not os.path.exists(directory):
    os.makedirs(directory)
    
  with open(path, 'w') as f:
    json.dump(data, f, default=np_encoder)


def get_data(summ_dls, summ_dataset):
  data = {}
  
  with torch.no_grad():
    for entity_id, entity_loader in (summ_dls.items()):
      data[entity_id] = {}
      data[entity_id]['texts'] = []

      texts = []
      representations = []
      sent_labels = [] # sentence-level review label

      for batch in entity_loader:
        src = batch[0].to(device)
        ids = batch[2]
        for full_id in ids:
            entity_id, review_id = full_id.split('__')
            text = summ_dataset.reviews[entity_id][review_id]
            data[entity_id]['texts'].extend(text)
            sent_labels.extend([summ_dataset.labels[entity_id][review_id]] * len(text))

        batch_size, nsent, ntokens = src.size()

        _, _, _, dist = model.cluster(src)

        representations.extend(dist)

      representations = torch.stack(representations)
      representations = representations.reshape(-1, 8 * 1024).detach().cpu().numpy()
      data[entity_id]['representations'] = representations
      data[entity_id]['rating'] = sent_labels
  return data


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



def run_online_summarization(points, summarizer=BFOnlineSummarizer()):
  import time # adhoc fix. TODO:find the root cause of this bug
  start = time.time()
  for i in (range(points.shape[0])):
    summ = summarizer.update_summary(points[i])
  return time.time() - start


def online_summary(points, summarizer=BFOnlineSummarizer()):
  all_summs = []
  for point in points:
    summ = summarizer.update_summary(point)
    if args.data_type == 'vector':
      all_summs.append(list(summ))
    elif args.data_type == 'text':
      all_summs.append(summ)
  return all_summs


def test_correctness(points, sg_summarizer = CoverSummOnlineSummarizer()):
  bf_summarizer = BFOnlineSummarizer()

  compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
  correct = 0
  total = 0

  for i in tqdm(range(points.shape[0])):
    bf_summ = bf_summarizer.update_summary(points[i])
    sg_summ = sg_summarizer.update_summary(points[i])
    match = compare(sg_summ, bf_summ)

    correct += match
    total += 1
  
  return correct / total


def load_splits(path='../../../data/space/space_summ_splits.txt'):
  split_dict = {}
  with open(path) as file:
    for line in file:
      entity_id, split = line.strip().split()
      split_dict[entity_id] = split
  return split_dict


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--summarizer",
                      default="coversumm",
                      type=str,
                      help="Name of Summarizer.")
  parser.add_argument("--max_iter",
                      default=1,
                      type=int,
                      help="Number of runs.")
  parser.add_argument('--model',
                      help='trained SemAE model to use',
                      type=str,
                      # default='../../../../models/space_checkpoint.pt')
                      default='../../../../third_party/SemAE/models/spacev2_7_model.pt')
  parser.add_argument('--summary_data',
                      help='summarization benchmark data',
                      type=str,
                      # default='../../../data/space/json/space_summ.json')
                      default='../../../../third_party/SemAE/data/space/json/space_summ.json')
  parser.add_argument('--out_dir',
                      help='output directory',
                      type=str,
                      default='../../../outputs/online_summaries/')
  parser.add_argument('--data_type',
                      help='input data type to the summarizer (text/vector)',
                      type=str,
                      default='vector')
  parser.add_argument(
      '--sentencepiece',
      help='sentencepiece model file',
      type=str,
      # default='../../../data/sentencepiece/spm_unigram_32k.model')
      default='../../../../third_party/SemAE/data/sentencepiece/spm_unigram_32k.model')
  parser.add_argument(
      '--gpu',
      help='gpu device to use (default: -1, i.e., use cpu)',
      type=int,
      default=0)
  parser.add_argument('--batch_size',
                        help='the maximum batch size (default: 5)',
                        type=int,
                        default=5)

  args = parser.parse_args()
  
  device = torch.device('cuda:{0}'.format(args.gpu))
  model = torch.load(args.model, map_location=device)
  
  data = load_dataset(args.summary_data, args.sentencepiece)
  splits = load_splits()

  import time

  total_time = 0
  num_entities = 0
  summaries = {}
  for i in range(args.max_iter):
    for entity_id, entity_data in tqdm(data.items()):
      if splits[entity_id] == 'dev':
        continue
      
      summarizer = get_summarizer(args.summarizer)
      if args.summarizer in ['textrank', 'lsa', 'sumbasic']:
        args.data_type = 'text'
      else:
        args.data_type = 'vector'

      if args.data_type == 'vector':
        representations = data[entity_id]['representations']
        points = representations.astype(dtype=np.float32)
        points = points / np.linalg.norm(points)
        start = time.time()
        summaries[entity_id] = online_summary(points, summarizer)
      elif args.data_type == 'text':
        texts = data[entity_id]['texts']
        start = time.time()
        summaries[entity_id] = online_summary(texts, summarizer)


      runtime = time.time() - start
      total_time += runtime
      num_entities += 1
      print(f"Iteration - {i}: Time - {runtime}")

  output_path = os.path.join(args.out_dir, args.summarizer, "summaries.json")
  dump_data(summaries, output_path)
  print(f"Amortized time - {total_time / args.max_iter / num_entities}")
