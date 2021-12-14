import codecs
from bs4 import BeautifulSoup
import numpy as np
import json
from pyserini.index import IndexReader 
from pyserini.search import SimpleSearcher
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Searcher(metaclass=Singleton):
    config = json.loads(open("config.json", "r").read())
    searcher = SimpleSearcher(config['index_path'])
    searcher.set_bm25(0.9, 0.4)
    
    @staticmethod
    def search(query: str, k: int) -> list:
        return Searcher.searcher.search(query, k);
    
class IndexReader(metaclass=Singleton):
    config = json.loads(open("config.json", "r").read())
    reader = IndexReader(config['index_path'])
    
    @staticmethod
    def analyze(query):
        return IndexReader.reader.analyze(query)

def get_similar_words(word : str, model, n : int = 10) -> list:
    if model.has_index_for(word):
        return model.most_similar(word, topn = n)
    else:
        return []
    
def get_similar_words_from_sentence(sentence: str, model, n: int):
    sentence = sentence.split()
    extra_words_and_weights = model.most_similar(positive = sentence, topn=n)
    extra_words = np.array(extra_words_and_weights)
    if np.size(extra_words)>0:
        flipped = 1-extra_words[:,1].astype(np.float64)
        extra_words = extra_words[np.argsort(flipped)][:n,0]
    return extra_words

def expand_query(topic, model, n = 10 ) -> str:
    ps = PorterStemmer()
    topic = word_tokenize(topic)
    clean_topic = []
    for word in topic:
        if word not in stopwords.words('english'):
            clean_topic.append(word.lower())
    stemmed_query = [ps.stem(w) for w in clean_topic]
    
    final_extra_words = []
    similar_words = []
    itr = 0
    idx = 0
    while(len(final_extra_words) < n):
        if idx >= len(similar_words):
            itr+=1
            similar_words = get_similar_words_from_sentence(' '.join(clean_topic), model, itr*n)
        
        if len(similar_words) > 0:
            word = similar_words[idx]
            if ps.stem(word) not in stemmed_query:
                final_extra_words.append(word)
                stemmed_query += ps.stem(word)
            if itr == n:
                break
            idx+=1
        else:
            break
    return ' '.join(topic + final_extra_words)
    
def parse_topic(topic_tag : str) -> dict:
    key = topic_tag.find('num').get_text().split()[1]
    value = ' '.join(topic_tag.find('title').get_text().split('\n')[0].split())
    return (key, value)

def get_topics(path : str) -> dict:
    topics_file = codecs.open(path, 'r', 'utf-8').read()
    soup = BeautifulSoup(topics_file, features='html.parser')
    topics = {}
    for t in soup.findAll('top'):
        kvp = parse_topic(t)
        topics[kvp[0]] = kvp[1]
    return topics

def parse_qrel_line(line):
    query, _, document, relevancy = line.split()
    query = int(query)
    relevancy = 1 if int(relevancy) >= 1 else 0
    return query, document, relevancy

def parse_results_line(line):
    query, _, document, rank, score, runid = line.split()
    query = int(query)
    rank = int(rank)
    score = float(score)
    return query, document, rank, score, runid

def query_labels_from_file(qrels_path, results_path):    
    relevancies = {}
    with open(qrels_path, 'r') as qrel_file:
        for line in qrel_file:
            query, document, relevancy = parse_qrel_line(line)
            if query not in relevancies:
                relevancies[query] = {document: relevancy}
            else:
                relevancies[query][document] = relevancy
            
    with open(results_path, 'r') as results_file:
        current_query, document, rank, _, _ = parse_results_line(next(results_file))
        label = relevancies[current_query][document]
        rank_label_list = [(rank, label)]
        for line in results_file:
            query, document, rank, _, _ = parse_results_line(line)
            if query != current_query:
                sorted_labels = [x[1] for x in sorted(rank_label_list)]
                yield np.array(sorted_labels, dtype=np.int32)
                current_query = query
                rank_label_list = []
            # TODO: experimental
            relevancy = relevancies[query]
            label = relevancy[document] if document in relevancy else 0
            # TODO: End experimental
            rank_label_list.append((rank, label))
        sorted_labels = [x[1] for x in sorted(rank_label_list)]
        yield np.array(sorted_labels, dtype=np.int32)
        
def DCG(query_relevancy_labels, k):
    y = query_relevancy_labels
    k = min(len(y), k)
    return np.sum( [(r_i / np.log2(i) ) for i, r_i in enumerate(y[:k], 2)] )
        
def NDCG(query_relevancy_labels, k):
    y = query_relevancy_labels
    k = min(len(y), k)
    idcg = DCG(np.sort(y)[::-1], k)
    return 0 if idcg == 0 else (DCG(y, k) / idcg)

def precision(query_relevancy_labels, k):
    y = query_relevancy_labels
    return np.sum(y[:k])/k

def MAP(query_relevancy_labels):
    y = query_relevancy_labels
    numerator = np.sum([y[k-1]*precision(y, k) for k in range(1, len(y)+1)])
    denomenator = 1 if np.sum(y) == 0 else np.sum(y)
    if denomenator == 0:
        result = 0
    else:
        result = numerator/denomenator
    return result
