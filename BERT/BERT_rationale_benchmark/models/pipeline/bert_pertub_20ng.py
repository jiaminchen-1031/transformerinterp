import argparse
import json
import logging
import random
import os

from sklearn.metrics import accuracy_score

from itertools import chain
from typing import List, Tuple
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator

from BERT_rationale_benchmark.utils import (
    Annotation,
    Evidence,
    write_jsonl,
    load_datasets,
    load_documents,
)
from BERT_explainability.modules.BERT.BertForSequenceClassification import \
    BertForSequenceClassification as BertForSequenceClassificationTest
from BERT_explainability.modules.BERT.BERT_cls_lrp import \
    BertForSequenceClassification as BertForClsOrigLrp

from transformers import BertForSequenceClassification

from collections import OrderedDict
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)
# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import numpy as np

latex_special_token = ["!@#$%^&*()"]


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list

def get_input_words(input, tokenizer, input_ids):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]

    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)

    start_idx = 0
    end_idx = 0
    words_from_chars = []
    for inp in input:
        if start_idx >= len(input_ids_chars):
            break
        end_idx = end_idx + len(inp)
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))
        start_idx = end_idx

    if (words_from_chars[:-1] != input[:len(words_from_chars)-1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False
    return words_from_chars

def bert_tokenize_doc(doc: List[List[str]], tokenizer, special_token_map) -> Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
    """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
    sents = []
    sent_token_spans = []
    for sent in doc:
        tokens = []
        spans = []
        start = 0
        for w in sent:
            if w in special_token_map:
                tokens.append(w)
            else:
                tokens.extend(tokenizer.tokenize(w))
            end = len(tokens)
            spans.append((start, end))
            start = end
        sents.append(tokens)
        sent_token_spans.append(spans)
    return sents, sent_token_spans

def initialize_models(params: dict, batch_first: bool, use_half_precision=False):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    bert_dir = params['bert_dir']
    evidence_classes = dict((y, x) for (x, y) in enumerate(params['evidence_classifier']['classes']))
    evidence_classifier = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=len(evidence_classes))
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens
    return evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer


BATCH_FIRST = True


def extract_docid_from_dataset_element(element):
    return next(iter(element.evidences))[0].docid

def extract_evidence_from_dataset_element(element):
    return next(iter(element.evidences))


def main():
    parser = argparse.ArgumentParser(description="""Trains a pipeline model.

    Step 1 is evidence identification, that is identify if a given sentence is evidence or not
    Step 2 is evidence classification, that is given an evidence sentence, classify the final outcome for the final task
     (e.g. sentiment or significance).

    These models should be separated into two separate steps, but at the moment:
    * prep data (load, intern documents, load json)
    * convert data for evidence identification - in the case of training data we take all the positives and sample some
      negatives
        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a
          broader sampling of negative values.
    * train evidence identification
    * convert data for evidence classification - take all rationales + decisions and use this as input
    * train evidence classification
    * decode first the evidence, then run classification for each split
    
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--neg', type=bool, default=False,
                        help='neg')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    parser.add_argument('--method', type=str, default='transformer_attribution', choices=["transformer_attribution", "partial_lrp", "last_attn",
                         "attn_gradcam", "rollout", "generic_attribution", "ours", "ground_truth", "generate_all", "ours_c"])
    args = parser.parse_args()
    assert BATCH_FIRST
    
    with open(args.model_params, 'r') as fp:
        logger.info(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
        logger.info(f'Params: {json.dumps(model_params, indent=2, sort_keys=True)}')
    
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame(data={"label":newsgroups_test["target"],"sentence":newsgroups_test["data"]})
    sentences = df.sentence.values
    labels = df.label.values
    tokenizer = BertTokenizer.from_pretrained('path-to-bert-base-uncased', do_lower_case=True)
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            max_length = 512,# Truncate all sentences.
                            pad_to_max_length = True
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    attention_masks = []
    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    test_input = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    # test
    device = torch.device("cuda")
    model_save_file = 'path-to-finetuned-model '
    test_classifier = BertForSequenceClassificationTest.from_pretrained(model_params['bert_dir'],
                                                                        num_labels=20).to(device)
    orig_lrp_classifier = BertForClsOrigLrp.from_pretrained(model_params['bert_dir'],
                                                            num_labels=20).to(device)
    if os.path.exists(model_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        test_classifier.load_state_dict(torch.load(model_save_file))
        orig_lrp_classifier.load_state_dict(torch.load(model_save_file))
        test_classifier.eval()
        orig_lrp_classifier.eval()
        test_batch_size = 1
        logging.info(
            f'Testing with {len(test_input) // test_batch_size} batches with {len(test_input)} examples')

        # explainability
        explanations = Generator(test_classifier)
        explanations_orig_lrp = Generator(orig_lrp_classifier)
        method = args.method
        method_folder = {"transformer_attribution": "transformer_attribution", "partial_lrp": "partial_lrp", "last_attn": "last_attn",
                         "attn_gradcam": "attn_gradcam", "rollout": "rollout", "generic_attribution":  "generic_attribution", "ours": "ours", "ours_c": "ours_c",  "ground_truth": "ground_truth", "generate_all": "generate_all"}
        method_expl = {"transformer_attribution": explanations_orig_lrp.generate_LRP,
                       "partial_lrp": explanations_orig_lrp.generate_LRP_last_layer,
                       "last_attn": explanations_orig_lrp.generate_attn_last_layer,
                       "attn_gradcam": explanations_orig_lrp.generate_attn_gradcam,
                       "rollout": explanations_orig_lrp.generate_rollout,
                      "ours": explanations.generate_ours,
                        "ours_c": explanations.generate_ours_c,
                      "generic_attribution": explanations_orig_lrp.generate_genattr}

        pertub_step = np.arange(0,1.1,0.1)
        pertub_acc = [0]*len(pertub_step)
        
        data = TensorDataset(test_input, test_masks, test_labels)
        num_samples = 3000
        import random
        random.seed(0)
        index = random.sample(range(len(data)), num_samples)
        sub_dataset = torch.utils.data.Subset(data, indices=index)
        dataloader = DataLoader(sub_dataset, batch_size=test_batch_size)
        
        iterator = tqdm(dataloader)

        for batch_idx, batch in enumerate(iterator):
            input_ids = batch[0].type(torch.LongTensor).to(device)
            attention_masks = batch[1].to(device)
            targets = batch[2].type(torch.LongTensor).to(device)
            preds = test_classifier(input_ids=input_ids, attention_mask=attention_masks)[0]

            classification = "neg" if targets.item() == 0 else "pos"
            is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0

            classification = "neg" if targets.item() == 0 else "pos"
            is_classification_correct = 1 if preds.argmax(dim=1) == targets else 0
            target_idx = targets.item()
            cam_target = method_expl[method](input_ids=input_ids, attention_mask=attention_masks, index=target_idx)[0]
            cam_target = cam_target.clamp(min=0)
            if args.neg:
                cam_target = -cam_target

            for step_idx, step in enumerate(pertub_step):
                num_token_perb = int(input_ids.shape[1]*step)
#                     cam = scores_per_word_from_scores_per_token(inp, tokenizer,input_ids[0], cam_target)
                _, indices = cam_target.topk(k=num_token_perb)
                input_ids[:, indices] = 0
                attention_masks[:, indices] = 0
                preds = test_classifier(input_ids=input_ids, attention_mask=attention_masks)[0]
                acc = 1 if preds.argmax(dim=1) == targets else 0
                pertub_acc[step_idx] += acc

            curr_pert_result = [round(res / (batch_idx+1) * 100, 2) for res in pertub_acc]
            iterator.set_description("Acc: {}".format(curr_pert_result))
                

if __name__ == '__main__':
    main()
