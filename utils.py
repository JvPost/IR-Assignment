import codecs
from bs4 import BeautifulSoup
import numpy as np

def get_similar_words(word : str, model, n : int = 10) -> list:
    if model.has_index_for(word):
        return model.most_similar(word, topn = n)
    else:
        return []
    
def expand_query(query, model, n = 10 ) -> str:
    query = query.lower().split()
    extra_words = []
    for q in query:
        extra_words += get_similar_words(q, model, n)
    extra_words = np.array(extra_words)
    if np.size(extra_words)>0:
        flipped = 1 - extra_words[:,1].astype(np.float64) # flip to make the argsort descending
        extra_words = extra_words[np.argsort(flipped)][:n,0]
    return ' '.join(query + list(extra_words))
    
def parse_topic(topic_tag : str) -> dict:
    key = topic_tag.find('num').get_text().split()[1]
    value = ' '.join(topic_tag.find('title').get_text().split('\n')[0].split())
    return (key, value)

def get_topics(path : str) -> dict:
    topics_file = codecs.open(path, 'r', 'utf-8').read()
    soup = BeautifulSoup(topics_file)
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


