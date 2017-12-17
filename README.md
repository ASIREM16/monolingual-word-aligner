# A Word Aligner for English

This is a word aligner for English: given two English sentences, it aligns related words in the two sentences. It exploits the semantic and contextual similarities of the words to make alignment decisions.


## Ack
Initially, this is a fork of <i>[ma-sultan/monolingual-word-aligner](https://github.com/ma-sultan/monolingual-word-aligner)</i>, the aligner presented in [Sultan et al., 2015](http://aclweb.org/anthology/S/S15/S15-2027.pdf) that has been very successful in [SemEval STS (Semantic Textual Similarity) Task](http://alt.qcri.org/semeval2017/task1/) in recent years.


## Install
```bash
# download the repo
git clone https://github.com/rgtjf/monolingual-word-aligner.git

# require stopwords from nltk
python -m nltk.downloader stopwords

# require stanford corenlp
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip
unzip stanford-corenlp-full-2015-12-09.zip

# lanch the stanford CoreNLP
cd stanford-corenlp-full-2015-12-09/
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
# after this, you will find stanfordCoreNLP server at http://localhost:9000/

python test_align.py
```

## Evaluate on STSBenchmark

```bash
sh download.sh
python run_stsbenchmark.py
```

### Results

| Methods (eval on STSbenchmark) | Dev    | Test   |
|--------------------------------|--------|--------|
| aligner                        | 0.6991 | 0.6379 |
| idf_aligner                    | 0.7969 | 0.7622 |


### Reference
[STSBenchmark board](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
