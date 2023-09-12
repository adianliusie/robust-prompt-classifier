# Robust Prompt-Based Classifier

This is the official implementation of our paper, [Mitigating Word Bias in Prompt-Based Classifiers](https://arxiv.org/pdf/2309.04992.pdf), which was accepted at IJNCLP-AACL 2023. 

> Authors: Adian Liusie, Potsawee Manakul, Mark J. F. Gales \
> Abstract: 
Prompt-based classifiers are an attractive approach for zero-shot classification. However, the precise choice of the prompt template and label words can largely influence performance, with semantically equivalent settings often showing notable performance difference. This discrepancy can be partly attributed to word biases, where the classifier may be biased towards classes. To address this problem, it is possible to optimise classification thresholds on a labelled data set, however, this mitigates some of the advantages of prompt-based classifiers. This paper instead approaches this problem by examining the expected marginal probabilities of the classes. Here, probabilities are reweighted to have a uniform prior over classes, in an unsupervised fashion. Further, we draw a theoretical connection between the class priors and the language models’ word prior, and offer the ability to set a threshold in a zero-resource fashion. We show that matching class priors correlates strongly with the oracle upper bound performance and demonstrate large consistent performance gains for prompt settings over a range of NLP tasks.

The repo allows one to select a prompt, class label words, and prompted language model, and enables one to generate more robust classification predictions

## Installation
```
pip install -r requirements.txt
```

## Running Experiments of the paper
### Step 1: generate model outputs
Automatic scripts provided are `search/nli.py` (nli), `search/qqp.py` (paraphrase), `search/sentimeny.py` (sentiment classification). Running the  experiment can be done e.g. as:
```
python search/sentiment.py --transformer flan-t5-large --path outputs/flan-t5-large/
```

Arguments:
- `transformer` which base LLM to use, one of [`flan-t5-base`, `flan-t5-large`, `llama-2-7b`]
- `path` output path, where to save the experiment output

Note that one can create your own custom script and define the **templates**, **label word sets** and **datasets** (dataset needs to be interfaced in `src/data/data_handler.py` within the `load_data` method) 

### Step 2: extract/analyze predicted class decisions
the `LogitsReader` class is used to process the output of the results. One can generate predictions at different levels, based on the available information

```
from analysis.logits_reader import LogitsReader

logits_reader = LogitsReader(path=path, dataset=dataset)

baseline_probs = logits_reader.load_probs()                          # baseline class probabilities (using basic likelihood
null_input_logits = logits_reader.load_probs(norm='null-norm')       # zero-resource calibrated probs, using implicit model prior

label_words = ['bad', 'amazing']                                                   # The search procedures require a selected label word set
prior_match_probs = logits_reader.balanced_alpha_logits(label_words=label_words)   # prior-matched probabilities (uses inputs but no labels)
optimal_probs = logits_reader.optimal_N_logits(label_words=label_words)            # optimal thresholds
```

Alternatively, all scripts used to generate the figures of the paper can be found in the anlaysis folder.
