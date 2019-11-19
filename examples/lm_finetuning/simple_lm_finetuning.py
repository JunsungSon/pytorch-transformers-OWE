# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import random
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import re
import pickle

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_bert import BertForPreTraining, BertKBForMaskedLM, BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from torch.nn import CrossEntropyLoss

from pathlib import Path

from owe import data
from owe.models import Mapper
from owe.config import Config
from owe.utils import read_config, load_checkpoint

from nltk.corpus import stopwords
#stopword_set = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    filename='simple_lm_fintuning.log')
logger = logging.getLogger(__name__)


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True, owe_vocab=None):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.stopwords = set(stopwords.words('english'))

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        self.owe_vocab = owe_vocab

        self.sents = []
        assert on_memory
        # load samples into memory
        if on_memory:
            self.corpus_lines = 0
            corpus_fnames = os.listdir(corpus_path)
            total_line_count = 0
            for fname in corpus_fnames:
                if ".pkl" not in fname:
                    continue
                fname = os.path.join(corpus_path, fname)
                finfo = os.stat(fname)
                if finfo.st_size < 100: # to drop incomplete files
                    continue

                with open(fname, "rb") as f:
                    d = pickle.load(f)
                    for elem in tqdm(d.values(), desc="Loading Dataset", total=len(d.keys())):
                        total_line_count += 1
                        # Validity check
                        # ================================
                        obj_flag, rel_flag = False, False
                        for t in elem['tags']:
                            if "B-" in t:
                                if "V" in t.split("-")[1]:
                                    rel_flag = True
                                else:
                                    obj_flag = True
                        if not (obj_flag and rel_flag):
                            continue
                        # ================================
                        self.sents.append(elem)
                        self.corpus_lines = self.corpus_lines + 1
            print()
            print("{} / {} is trinable resource.".format(self.corpus_lines, total_line_count))
        # load samples later lazily from disk
        else:
            print("on memory argument should be used")
            exit(-1) # on memory만 쓸거니까 막늗나.
            if self.corpus_lines is None:
                with open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        return len(self.sents)

    def parse(self, elem):
        tokens, tags = elem['tokens'], elem['tags']
        text = ' '.join(tokens)

        obj_lists = []
        rel_lists = []
        char_offset = 0
        tmp = []
        tag_flag = None
        for tok, tag in zip(tokens, tags):
            if "B-" in tag:
                if len(tmp) > 0 and tag_flag is not None:
                    tmp.append(char_offset - 1)
                    if "V" in tag_flag:
                        rel_lists.append(tmp)
                    else:
                        obj_lists.append(tmp)
                tag_flag = tag.split("-")[1]
                tmp = [char_offset]
            elif "I-" in tag:
                tmp.append(char_offset)
            char_offset += len(tok)
        # 마지막 토큰
        if len(tmp) > 0:
            if "V" in tag_flag:
                rel_lists.append(tmp)
            else:
                obj_lists.append(tmp)

        rel_offsets = []
        obj_offsets = []
        for r_e in rel_lists:
            rel_offsets.append([r_e[0], r_e[-1]])
        for o_e in obj_lists:
            obj_offsets.append([o_e[0], o_e[-1]])
        if len(rel_lists) == 0:
            print("[ERROR case in 'parse' function']")
            print(tokens)
            print(tags)
            exit(-1)
        return text, obj_offsets, rel_offsets

    def __getitem__(self, item):
        def tokenize_old(content, lower=True, remove_punctuation=True, add_underscores=False, limit_len=100000):
            """
            Splits on spaces between tokens.

            :param content: The string that shall be tokenized.
            :param lower: Lowers content string
            :param remove_punctuation: Removes single punctuation tokens
            :param add_underscores: Replaces spaces with underscores
            :return:
            """
            import re
            from nltk.tokenize import word_tokenize

            if not content or not limit_len:
                return [""] if add_underscores else []

            if not isinstance(content, (str)):
                raise ValueError("Content must be a string.")

            if remove_punctuation:
                content = re.sub('[^A-Za-z0-9 ]+', '', content)

            if lower:
                content = content.lower()

            if add_underscores:
                res = [re.sub(' ', '_', content)]
                return res

            res = word_tokenize(content)
            return res

        cur_id = self.sample_counter
        self.sample_counter += 1
        assert self.on_memory
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)

        text, obj_offsets, rel_offsets = self.parse(self.sents[item])

        # tokenize
        tokens = self.tokenizer.tokenize(text)
        tokens_owe = tokenize_old(text)

        # combine to one sample
        cur_example = InputExample(guid=cur_id,
                                   text=text,
                                   tokens=tokens,
                                   tokens_owe=tokens_owe,
                                   obj_offsets=obj_offsets,
                                   rel_offsets=rel_offsets)

        with open("token2id.json", "r+", encoding='utf-8') as tmp_f:
            import json
            owe_vocab = json.load(tmp_f)
        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer, owe_vocab)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_owe_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.mask_label_ids),
                       torch.tensor(cur_features.arg_label_mask),
                       )

        return cur_tensors


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, tokens, tokens_owe=None, obj_offsets=None, rel_offsets=None, text=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.tokens = tokens
        self.tokens_owe = tokens_owe
        self.obj_offsets = obj_offsets
        self.rel_offsets = rel_offsets
        self.text = text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_owe_ids, input_mask, segment_ids, mask_label_ids, arg_label_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mask_label_ids = mask_label_ids
        self.input_owe_ids = input_owe_ids
        self.arg_label_mask = arg_label_mask


def masking_word(tokens, tokenizer, obj_offsets, rel_offset):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    arg_mask = []
    rel_start, rel_end = rel_offset[0], rel_offset[1]
    cur_offset = 0
    # (h,r,t)에서 R 만 마스크를 뚫어야 관계학습이 가능함
    for i, token in enumerate(tokens):
        prob = random.random()
        # 이 토큰이 Relation이면 마스킹한다
        if rel_start <= cur_offset <= rel_end:
            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

        obj_flag = False
        for obj_offset in obj_offsets:
            if obj_flag:
                break
            obj_start, obj_end = obj_offset[0], obj_offset[1]
            if obj_start <= cur_offset <= obj_end:
                arg_mask.append(1)
                obj_flag = True
        if obj_flag == False:
            arg_mask.append(-1)
        cur_offset += max(1, len(token.replace("#", ""))) # At least word is longer or equal than 1

    assert len(tokens) == len(arg_mask), "[masking_word] function, arg_mask Error"
    assert len(tokens) == len(output_label), "[masking_word] function, output_label Error"
    return tokens, output_label, arg_mask


def convert_example_to_features(example, max_seq_length, tokenizer, owe_vocab):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    text = example.text
    tokens = example.tokens
    tokens_owe = example.tokens_owe
    obj_offsets = example.obj_offsets
    rel_offset = example.rel_offsets[0]

    tokens = tokens[:max_seq_length - 2]
    tokens_owe = tokens_owe[:max_seq_length - 2]
    tokens, mask_label, arg_mask = masking_word(tokens, tokenizer, obj_offsets, rel_offset)
    lm_label_ids = ([-1] + mask_label + [-1])
    arg_label_mask = ([-1] + arg_mask + [-1])
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    _tokens = []
    segment_ids = []
    _tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens:
        _tokens.append(token)
        segment_ids.append(0)
    _tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(_tokens)
    input_owe_ids = [owe_vocab[w] if w in owe_vocab else owe_vocab["_UNK_"] for w in tokens_owe]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)
        arg_label_mask.append(-1)
    input_owe_ids = input_owe_ids + [0] * (max_seq_length - len(input_owe_ids))

    assert len(input_ids) == max_seq_length
    assert len(input_owe_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length
    assert len(arg_label_mask) == max_seq_length

    '''
    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))
    '''
    features = InputFeatures(input_ids=input_ids,
                             input_owe_ids=input_owe_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             mask_label_ids=lm_label_ids,
                             arg_label_mask=arg_label_mask
                             )
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=2048,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", 
                        default=0, 
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=64,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--output_attentions',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")


    # OWE args
    parser.add_argument('-c', '--corpus_directory', help="Directory where the dataset is stored", required=True)
    parser.add_argument('-d', '--output_directory', help="Where to output artifacts.", required=True)
    model_parser = parser.add_mutually_exclusive_group(required=True)
    model_parser.add_argument('--distmult', help="Load a pretrained DistMult model from given directory.")
    model_parser.add_argument('--transe', help="Load a pretrained TransE model from given directory.")
    model_parser.add_argument('--complex', help="Load a pretrained ComplEx model from given directory.")
    model_parser.add_argument('--rotate', help="Load a pretrained RotatE model from given directory.")
    checkpoint_parser = parser.add_mutually_exclusive_group()
    checkpoint_parser.add_argument('-l', '--load', help="Load the last OWE model checkpoint if one exists.",
                                   action='store_true')
    checkpoint_parser.add_argument('-lb', '--load_best', help="Load the best OWE model checkpoint if one exists.",
                                   action='store_true')

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    ################### OWE
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    # Try to load config file
    config_file = output_directory / "config.ini"
    if not config_file.exists():
        raise FileNotFoundError("No config file found under: {}.".format(config_file))
    train_file, valid_file, test_file, skip_header, split_symbol, wiki_file = read_config(config_file)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # load KGC model
    kgc_model_name, kgc_model_directory = None, None
    if args.complex:
        kgc_model_name, kgc_model_directory = "ComplEx", Path(args.complex)
    elif args.transe:
        kgc_model_name, kgc_model_directory = "TransE", Path(args.transe)
    elif args.distmult:
        kgc_model_name, kgc_model_directory = "DistMult", Path(args.distmult)
    elif args.rotate:
        kgc_model_name, kgc_model_directory = "RotatE", Path(args.rotate)
        if args.gamma is None:
            raise ValueError("Specify gamma value from trained RotatE")

    Config.set("LinkPredictionModelType", kgc_model_name)
    logger.info("LinkPredictionModelType: {}".format(kgc_model_name))

    mapper = Mapper(torch.zeros((18804, 300), dtype=torch.float32), 300)
    mapper = torch.nn.DataParallel(mapper)
    #mapper = mapper.to(device)
    assert args.load_best, "You should input `load_best` in your command."
    if args.load_best:
        checkpoint_file = 'best_checkpoint.OWE.pth.tar'
        checkpoint_owe = load_checkpoint(output_directory / checkpoint_file)
        if checkpoint_owe:
            start_epoch = checkpoint_owe['epoch'] + 1
            mapper.load_state_dict(checkpoint_owe["mapper"], strict=False)
            logger.info("Initialized OWE model, mapper and optimizer from the loaded checkpoint.")

        del checkpoint_owe
    ########################


    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and ( n_gpu > 1 and torch.distributed.get_rank() == 0  or n_gpu <=1 ):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_corpus)
        train_dataset = BERTDataset(args.train_corpus, tokenizer, seq_len=args.max_seq_length,
                                    corpus_lines=None, on_memory=args.on_memory)
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    #model = BertForPreTraining.from_pretrained(args.bert_model)
    model = BertKBForMaskedLM.from_pretrained(args.bert_model, output_attentions=args.output_attentions)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())

        # We should freeze mapper(from OWE)'s parameters
        param_optimizer = [n for n in param_optimizer if 'mapper' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    global_step = 0

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        no_mask_sample_count = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_owe_ids, input_mask, segment_ids, lm_label_ids, arg_label_mask = batch
                outputs = model(input_ids, input_owe_ids, segment_ids, input_mask)
                with torch.no_grad():
                    gt_owe_vec = mapper(input_owe_ids, None)
                arg_label_mask = arg_label_mask.float() # 요 label은 hinge loss로 학습에 사용한다...
                # Loss를 여기서 계산한다.
                if args.output_attentions: # Relation Attention Loss
                    # NOTE : mask label을 채우기 위해 사용된 attention은 head, tail 이 가장 높아야 한다
                    prediction_scores, prediction_owe_vectors, output_attentions = outputs
                    output_attentions = torch.cat(output_attentions, dim=1) # N X 144 X src X tgt
                    # N X 1 X 1 X tgt => N X 144 X src X tgt
                    mask_label_ids_filter = torch.gt(lm_label_ids, 0).unsqueeze(1).unsqueeze(2).expand(-1, 144, lm_label_ids.size(1), -1).float()
                    try:
                        output_attentions = output_attentions * mask_label_ids_filter # N X 144 X src X tgt
                        output_attentions, _ = output_attentions.max(dim=1) # N X src X tgt
                        output_attentions = torch.clamp(output_attentions, 0.0, 1.0)
                        relation_loss = 1e+10 * (output_attentions * arg_label_mask.unsqueeze(1).expand(-1, lm_label_ids.size(1), -1)) # N X src X tgt
                        relation_loss = torch.sigmoid(relation_loss)
                        relation_loss = relation_loss.mean()
                    except Exception as e:
                        # No mask in this instance.
                        print("[Exception in this mini-batch] : \t", e)
                        relation_loss = 0.0
                        no_mask_sample_count += 1
                else:
                    prediction_scores, prediction_owe_vectors = outputs
                    relation_loss = 0.0

                loss_fct = CrossEntropyLoss(ignore_index=-1)
                masked_lm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
                loss = masked_lm_loss + relation_loss
                imitation_loss = ((gt_owe_vec - prediction_owe_vectors)**2).mean()
                loss += imitation_loss

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if args.output_attentions and nb_tr_steps < 100:
                    logger.info("Mask loss : {} | Imitating Loss : {} | Attention Supervision Loss : {}".format(masked_lm_loss, imitation_loss, relation_loss))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if nb_tr_steps % 5000 == 0:
                    # save
                    logger.info("[{}]** ** * Saving fine - tuned model ** ** * ".format(nb_tr_steps))
                    logger.info("Avg loss : {}".format(tr_loss / nb_tr_steps))
                    logger.info("Last loss : {}".format(loss))
                    logger.info("Last[masked_lm_loss] : {}".format(masked_lm_loss))
                    if args.output_attentions:
                        logger.info("Last[relation_loss] : {}".format(relation_loss))
                    logger.info("Last[imitation_loss] : {}".format(imitation_loss))
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)

        # Save a trained model
        if args.do_train and ( n_gpu > 1 and torch.distributed.get_rank() == 0  or n_gpu <=1):
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
