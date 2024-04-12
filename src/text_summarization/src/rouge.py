import json
import nltk
import argparse

from utils.data import *
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.summary import truncate_summary, RougeEvaluator

def load_data(path):
  with open(path, 'r') as f:
    return json.load(f)


def dump_json(content, filename):
    with open(filename, 'w') as file:
        json.dump(content, file)


def load_splits(path='../../../../summarization/quantization/qt/data/space_summ_splits.txt'):
  split_dict = {}
  with open(path) as file:
    for line in file:
      entity_id, split = line.strip().split()
      split_dict[entity_id] = split
  return split_dict

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
        ids = batch[2]
        for full_id in ids:
          entity_id, review_id = full_id.split('__')
          text = summ_dataset.reviews[entity_id][review_id]
          data[entity_id]['texts'].extend(text)
          sent_labels.extend([summ_dataset.labels[entity_id][review_id]] * len(text))
      data[entity_id]['rating'] = sent_labels
  return data


def get_ROUGE(ranked_entity_sentences):
  cos_thres = 0.75
  min_tokens = 2
  max_tokens = 75
  no_cut_sents = True
  no_early_stop = True

  for entity_id, ranked_sentences in ranked_entity_sentences.items():
    if splits[entity_id] == 'test':
      file_path = os.path.join(output_path, 'test_' + entity_id)

      summary_sentences = truncate_summary(ranked_sentences,
              max_tokens=max_tokens, cut_sents=(not no_cut_sents),
              vectorizer=vectorizer, cosine_threshold=cos_thres,
              early_stop=(not no_early_stop),
              min_tokens=min_tokens)

      fout = open(file_path, 'w')
      fout.write(delim.join(summary_sentences))
      fout.close()
  
  # evaluate summaries
  sfp = '(.*)'
  mfp = '#ID#_[012].txt'
  gold_path = '../data/space/gold'

  test_evaluator = RougeEvaluator(system_dir=output_path,
                              system_filename_pattern='test_' + sfp,
                              model_filename_pattern=mfp)

  dict_results = {'dev':{}, 'test':{}, 'all':{}}


  model_dir = os.path.join(gold_path, 'general')
  outputs = test_evaluator.evaluate(model_dir=model_dir)
  dict_results['test']['general'] = outputs['dict_output']
  
  R1 = dict_results['test']['general']['rouge_1_f_score']
  R2 = dict_results['test']['general']['rouge_2_f_score']
  RL = dict_results['test']['general']['rouge_l_f_score']
  return R1, R2, RL

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--summarizer",
                      default="textrank",
                      type=str,
                      help="Name of Summarizer.")
  parser.add_argument('--summary_data',
                      help='summarization benchmark data',
                      type=str,
                      default='../../../../third_party/SemAE/data/space/json/space_summ.json')
  parser.add_argument('--data_type',
                      help='output type of summaries',
                      type=str,
                      default='int')
  parser.add_argument(
      '--sentencepiece',
      help='sentencepiece model file',
      type=str,
      default='../../../../third_party/SemAE/data/sentencepiece/spm_unigram_32k.model')
  parser.add_argument('--batch_size',
                        help='the maximum batch size (default: 5)',
                        type=int,
                        default=5)

  args = parser.parse_args()

  if args.summarizer in ['textrank', 'sumbasic', 'lsa']:
    args.data_type = 'text'
  else:
    args.data_type = 'int'
  
  data = load_dataset(args.summary_data, args.sentencepiece)
  summaries = load_data(f"../outputs/online_summaries/{args.summarizer}/summaries.json")


  all_texts = []
  for entity_id, summary in summaries.items():
    summary_ids = summary[-1]
    all_texts.extend(data[entity_id]['texts'])

  vectorizer = TfidfVectorizer(decode_error='replace', stop_words='english')
  vectorizer.fit(all_texts)

  # write summaries
  output_path = f"../outputs/space/{args.summarizer}/"
  os.makedirs(output_path, exist_ok=True)
  delim = '\t'

  splits = load_splits()

  k = 20 # minimum number of samples

  rouge_scores = {}
  for entity_id, summary in tqdm(summaries.items()):
    if splits[entity_id] == 'dev':
      continue
    
    rouge_scores[entity_id] = {}
    ranked_entity_sentences = {}
    for idx in tqdm(range(20, len(summary)+1, 20)):
      summary_ids = summary[idx]
      if args.data_type == 'int':
        ranked_entity_sentences[entity_id] = [data[entity_id]['texts'][i] for i in summary_ids]
      elif args.data_type == 'text':
        if args.summarizer == 'textrank':
          summary_ids = nltk.sent_tokenize(summary_ids)
        ranked_entity_sentences[entity_id] = summary_ids
      R1, R2, RL = get_ROUGE(ranked_entity_sentences)
      rouge_scores[entity_id][idx] = (R1, R2, RL)
  
  dump_json(rouge_scores, f"../outputs/space/rouge_scores_{args.summarizer}.json")
