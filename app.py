from utils import *
import json
import torch

config = json.loads(open("config.json", "r").read())
index_path = config["index_path"]
topics_path = config["topics_path"]
qrels_path = config["qrels_path"]
device = torch.device('cpu')



topics = get_topics(topics_path)
for topic in list(topics.values())[0:10]:
    print(expand_query(topic, wiki_300))