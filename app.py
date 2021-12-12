from utils import *
import json
import torch

config = json.loads(open("config.json", "r").read())
index_path = config["index_path"]
topics_path = config["topics_path"]
qrels_path = config["qrels_path"]
device = torch.device('cpu')


labels = query_labels_from_file(qrels_path, 'results.txt')
for l in labels:
    print(l)