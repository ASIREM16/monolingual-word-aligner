# coding: utf8
import codecs
import scipy.stats as meas
import pyprind

from word_aligner.corenlp_utils import StanfordNLP
from word_aligner.aligner import align_feats

nlp = StanfordNLP(server_url='http://localhost:9000')

def load_data(file_path):
    """load data"""
    with codecs.open(file_path, encoding='utf8') as f:
        data = []
        for line in f:
            line = line.strip().split('\t')
            score = float(line[4])
            sa, sb = line[5], line[6]
            data.append((sa, sb, score))
    return data


def aligner(data):
    preds = []
    golds = []
    process_bar = pyprind.ProgPercent(len(data))
    for example in data:
        process_bar.update()
        sa, sb, score = example
        parse_sa, parse_sb = nlp.parse(sa), nlp.parse(sb)
        features, infos = align_feats(parse_sa, parse_sb)
        preds.append(features[0])
        golds.append(score)
    return preds, golds


def evaluation(predict, gold):
    """
    pearsonr of predict and gold
    Args:
        predict: list
        gold: list
    Returns:
        pearsonr
    """
    pearsonr = meas.pearsonr(predict, gold)[0]
    return pearsonr

if __name__ == '__main__':
    test_file = './data/stsbenchmark/sts-test.csv'
    test_data = load_data(test_file)
    preds, golds = aligner(test_data)
    pearsonr = evaluation(preds, golds)
    print(pearsonr)

    # raw
    # sts-test 0.637979017879