# coding: utf8
from __future__ import print_function

import codecs
from collections import Counter

import math
import scipy.stats as meas
import pyprind
import argparse
import pickle
import os

from word_aligner.corenlp_utils import StanfordNLP
from word_aligner.aligner import align_feats


nlp = StanfordNLP(server_url='http://localhost:9000')


def load_data(file_path):
    """load data"""
    print('load the data from %s' % (file_path))
    with codecs.open(file_path, encoding='utf8') as f:
        data = []
        for line in f:
            line = line.strip().split('\t')
            score = float(line[4])
            sa, sb = line[5], line[6]
            data.append((sa, sb, score))
    return data


def parse_data(data):
    """parse data
    Returns:
        outputs: list of (parse_sa, parse_sb, score)
    """
    outputs = []
    process_bar = pyprind.ProgPercent(len(data), title='parse the data')
    for example in data:
        process_bar.update()
        sa, sb, score = example
        parse_sa, parse_sb = nlp.parse(sa), nlp.parse(sb)
        outputs.append((parse_sa, parse_sb, score))
    return outputs


def aligner(parsed_data):
    """aligner
        sim(sa, sb) = \frac{sa_{aligned} + sb_{aligned}}{sa_{all} + sb_{all}}
    """
    preds = []
    golds = []
    process_bar = pyprind.ProgPercent(len(parsed_data))
    for example in parsed_data:
        process_bar.update()
        parse_sa, parse_sb, score = example
        features, infos = align_feats(parse_sa, parse_sb)
        preds.append(features[0])
        golds.append(score)
    return preds, golds


def idf_aligner(parsed_data):
    """idf_aligner
        sim(sa, sb) = \frac{idf * sa_{aligned} + idf * sb_{aligned}}{idf * sa_{all} + idf * sb_{all}}
    """
    # obtain the idf_weight_dict
    sents = []
    for example in parsed_data:
        parse_sa, parse_sb, score = example
        sents.append(extract_words(parse_sa, 'lemma'))
        sents.append(extract_words(parse_sb, 'lemma'))
    idf_weight = idf_calculator(sents)
    min_idf_weight = min(idf_weight.values())

    # calculate the weighted alignment score
    preds = []
    golds = []
    process_bar = pyprind.ProgPercent(len(parsed_data))
    for example in parsed_data:
        process_bar.update()
        parse_sa, parse_sb, score = example
        features, infos = align_feats(parse_sa, parse_sb)
        # obtain the alignment information
        myWordAlignments = infos[0]
        # obtain the sents
        lemma_sa = extract_words(parse_sa, 'lemma')
        lemma_sb = extract_words(parse_sb, 'lemma')
        # the index of aligned words
        aligned_sa_idx = [sa_idx - 1 for sa_idx, sb_idx in myWordAlignments]
        aligned_sb_idx = [sb_idx - 1 for sa_idx, sb_idx in myWordAlignments]
        # binary representation
        aligned_sa = [0] * len(lemma_sa)
        aligned_sb = [0] * len(lemma_sb)
        for sa_index in aligned_sa_idx:
            aligned_sa[sa_index] = 1
        for sb_index in aligned_sb_idx:
            aligned_sb[sb_index] = 1
        # calc all and aligned except stopwords
        sa_sum = 0
        sb_sum = 0
        aligned_sa_sum = 0
        aligned_sb_sum = 0
        for idx, word in enumerate(lemma_sa):
            weight = idf_weight.get(word, min_idf_weight)
            sa_sum += weight
            aligned_sa_sum += aligned_sa[idx] * weight

        for idx, word in enumerate(lemma_sb):
            weight = idf_weight.get(word, min_idf_weight)
            sb_sum += weight
            aligned_sb_sum += aligned_sb[idx] * weight
        feature = [1.0 * (aligned_sa_sum + aligned_sb_sum) / (sa_sum + sb_sum + 1e-6)]

        preds.append(feature[0])
        golds.append(score)
    return preds, golds


def evaluation(predict, gold):
    """
    pearsonr of predict and gold
    Args:
        predict: list
        gold: list
    Returns:
        pearson of predict and gold
    """
    pearsonr = meas.pearsonr(predict, gold)[0]
    return pearsonr


def idf_calculator(sentence_list, min_cnt=1):
    doc_num = 0
    word_list = []
    for sequence in sentence_list:
        word_list += sequence
        doc_num += 1

    word_count = Counter()
    for word in word_list:
        word_count[word] += 1

    idf_dict = {}
    good_keys = [v for v in word_count.keys() if word_count[v] >= min_cnt]

    for key in good_keys:
        idf_dict[key] = word_count[key]

    for key in idf_dict.keys():
        idf_dict[key] = math.log(float(doc_num) / float(idf_dict[key])) / math.log(10)

    return idf_dict


def extract_words(parse_sent, type='word'):
    """
    type: 'word'/'lemma'/'pos'/'ner'
    """
    words = [token[type] for token in parse_sent['sentences'][0]['tokens']]
    return words


if __name__ == '__main__':

    test_file = './data/stsbenchmark/sts-test.csv'
    parsed_test_file = './data/stsbenchmark/sts-test.parse.pkl'

    # if not parse the data
    if not os.path.isfile(parsed_test_file):
        test_data = load_data(test_file)
        parsed_test_data = parse_data(test_data)
        pickle.dump(parsed_test_data, open(parsed_test_file, 'wb'), 2)

    parsed_test_data = pickle.load(open(parsed_test_file, 'rb'))

    # raw_aligner
    preds, golds = aligner(parsed_test_data)
    pearsonr = evaluation(preds, golds)
    print(pearsonr)  # stsbenchmark-test aligner 0.6379

    # idf_aligner
    preds, golds = idf_aligner(parsed_test_data)
    pearsonr = evaluation(preds, golds)
    print(pearsonr)  # stsbenchmark-test idf_aligner 0.7622
