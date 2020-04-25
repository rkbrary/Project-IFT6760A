import pickle
import random
import math
from collections import defaultdict

class Data:

    def __init__(self, data_dir="data/FB15k-237/", reverse=True, subset_percentage=0.5):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        redic = defaultdict(list)
        for triple in self.train_data:
            redic[(triple[1])].append(triple)

        for key in redic:
            redic[key] = random.sample(redic[key], math.floor(len(redic[key])*subset_percentage))
        bla=[]
        for key in redic:
            bla=bla+redic[key]
        self.train_data=bla

        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = sorted(list(set(self.train_relations + self.valid_relations + self.test_relations)))

    def load_data(self, data_dir, data_type="train", reverse=True):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
