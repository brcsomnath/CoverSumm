# Incremental Extractive Opinion Summarization Using Cover Trees

[![License: MIT](https://img.shields.io/badge/License-MIT-green``.svg)](https://opensource.org/licenses/MIT)

We present the implementation of the TMLR 2024 paper:

> [**Incremental Extractive Opinion Summarization Using Cover Trees**](https://arxiv.org/pdf/2401.08047),<br/>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/)<sup>1</sup>, [Nicholas Monath](https://people.cs.umass.edu/~nmonath/)<sup>2</sup>, [Kumar Avinava Dubey](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, [Manzil Zaheer](https://scholar.google.com/citations?hl=en&user=A33FhJMAAAAJ)<sup>2</sup>, [Andrew McCallum](https://people.cs.umass.edu/~mccallum/)<sup>2</sup>, [Amr Ahmed](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)<sup>1</sup>. <br>
UNC Chapel Hill<sup>1</sup>,  Google Deepmind<sup>2</sup>, Google Research<sup>3</sup>

## Overview

Extractive opinion summarization involves automatically producing a summary of text about an entity (e.g., a product’s reviews) by extracting representative sentences that capture prevalent opinions in the review set. Typically, in online marketplaces user reviews accrue over time, and opinion summaries need to be updated periodically to provide customers with up-to-date information. In this work, we study the task of extractive opinion summarization in an incremental setting, where the underlying review set evolves over time. Many of the state-of-the-art extractive opinion summarization approaches are centrality-based, such as CentroidRank (Radev et al., 2004; Chowdhury et al., 2022). CentroidRank performs extractive summarization by selecting a subset of review sentences closest to the centroid in the representation space as the summary. However, these methods are not capable of operating efficiently in an incremental setting, where reviews arrive one at a time. In this paper, we present an efficient algorithm for accurately computing the CentroidRank summaries in an incremental setting. Our approach, CoverSumm, relies on indexing review representations in a cover tree and maintaining a reservoir of candidate summary review sentences. CoverSumm’s efficacy is supported by a theoretical and empirical analysis of running time. Empirically, on a diverse collection of data (both real and synthetically created to illustrate scaling considerations), we demonstrate that CoverSumm is up to 36x faster than baseline methods, and capable of adapting to nuanced changes in data distribution. We also conduct human evaluations of the generated summaries and find that CoverSumm is capable of producing informative summaries consistent with the underlying review set.


![alt text](https://github.com/brcsomnath/CoverSumm/blob/master/data/intro_figure.png?raw=true)

## Installation
The simplest way to run our code is to start with a fresh environment.
```
conda create -n coverSumm python=3.6.13
source activate coverSumm
pip install -r requirements.txt
```

## Summarization Algorithms

The different algorithms used in the paper are available in the `src/algorithms/' folder. The detailed described of the file names and acronym to be used to run the algorithms can be found [here](src/algorithms/README.md). 



### Synthetic Data

The incremental summarization algorithms can be executed for synthetic data (Uniform & LDA distributions) using the following commands. To get the runtime scores for different algorithms use the following:

```
cd src/synthetic_summarization/
python launch.py \
        --summarizer <algorithm_name> \
        --distr <'uniform'/'lda'> \
        --num_samples 10000
```

To get the accuracy scores for nearest neighbour overlap of different algorithms use the following:

```
cd src/synthetic_summarization/
python correctness.py \
        --summarizer <algorithm_name> \
        --distr <'uniform'/'lda'> \
        --num_samples 10000
```

### SPACE

For running the algorithms on SPACE dataset, you would require access to the dataset ([link](https://github.com/stangelid/qt/)). You would also need a checkpoint of SemAE, which can be generated from [here](https://github.com/brcsomnath/SemAE/) or you can download the model used in our experiments directly [here](https://drive.google.com/file/d/12WRp7y_a-GiG8z4gP-_tJIuuRP-8qq6Q/view?usp=sharing). Place the generated or downloaded model in the `models/` folder. You can download the sentencepiece file from [here](https://github.com/stangelid/qt/tree/main/data/sentencepiece) and place it in the `data/sentencepiece/` folder.

```
cd src/text_summarization/src/
python launch_space.py \
        --summarizer <algorithm_name: coversumm> \
        --model '../../../models/space_checkpoint.pt' \
        --sentencepiece '../../../data/sentencepiece/spm_unigram_32k.model' \
```

### Amazon

The Amazon reviews dataset used in our experiments can be generated using the following command.

```
cd data/amazon/
python generate.py
```


Algorithms on Amazon use the BERT representations. You can directly run the following command:

```
cd src/text_summarization/src/
python launch_amazon.py \
        --summarizer <algorithm_name: coversumm> \
        --model 'bert-base-uncased' \
```

## Hyperparameters

More details about the hyperparameters coming soon.

## Reference

```
@article{
        chowdhury2024incremental,
        title={Incremental Extractive Opinion Summarization Using Cover Trees},
        author={Somnath Basu Roy Chowdhury and 
                Nicholas Monath and 
                Kumar Avinava Dubey and 
                Manzil Zaheer and
                Andrew McCallum and
                Amr Ahmed and 
                Snigdha Chaturvedi
        },
        journal={Transactions on Machine Learning Research},
        year={2024},
        url={https://openreview.net/forum?id=IzmLJ1t49R},
}
```
