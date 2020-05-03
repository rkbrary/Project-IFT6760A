import pickle
import random
import math
from collections import defaultdict

class Data:
#1= sym, 2, antisym, 3 other, 4 none
    def __init__(self, data_dir="/content/drive/My Drive/Project-IFT6760A-master/Project-IFT6760A-master/TuckER-master/data/WN18RR/", reverse=True, subset_percentage=0.5, asymorsymorother=1):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        if (asymorsymorother==1):
            sym=['_derivationally_related_form', '_similar_to', '_verb_group']
            new_train=[(a,r,b) for (a,r,b) in self.train_data if r in sym]#or (b,r,a) in self.train_data and b<=a
            self.train_data= new_train
        elif (asymorsymorother==2):
            asym=['_has_part', '_instance_hypernym', '_member_meronym','_member_of_domain_region','_member_of_domain_usage']
            new_train=[(a,r,b) for (a,r,b) in self.train_data if r in asym]#or (b,r,a) in self.train_data and b<=a
            self.train_data= new_train
        elif (asymorsymorother==3):
            others=['_also_see', '_hypernym', '_synset_domain_topic_of']
            new_train=[(a,r,b) for (a,r,b) in self.train_data if r in others]#or (b,r,a) in self.train_data and b<=a
            self.train_data= new_train
        else:
            print("Training with all relation types")

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
        if (asymorsymorother==1):
            sym=['_derivationally_related_form', '_similar_to', '_verb_group']
            new_val=[(a,r,b) for (a,r,b) in self.valid_data if r in sym]#or (b,r,a) in self.train_data and b<=a
            self.valid_data= new_val
        elif (asymorsymorother==2):
            asym=['_has_part', '_instance_hypernym', '_member_meronym','_member_of_domain_region','_member_of_domain_usage']
            new_val=[(a,r,b) for (a,r,b) in self.valid_data if r in asym]#or (b,r,a) in self.train_data and b<=a
            self.valid_data= new_val
        elif (asymorsymorother==3):
            others=['_also_see', '_hypernym', '_synset_domain_topic_of']
            new_val=[(a,r,b) for (a,r,b) in self.valid_data if r in others]#or (b,r,a) in self.train_data and b<=a
            self.valid_data= new_val
        else:
            print("Validating with all relation types")

        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        if (asymorsymorother==1):
            sym=['_derivationally_related_form', '_similar_to', '_verb_group']
            new_test=[(a,r,b) for (a,r,b) in self.test_data if r in sym]#or (b,r,a) in self.train_data and b<=a
            self.test_data= new_test
        elif (asymorsymorother==2):
            asym=['_has_part', '_instance_hypernym', '_member_meronym','_member_of_domain_region','_member_of_domain_usage']
            new_test=[(a,r,b) for (a,r,b) in self.test_data if r in asym]#or (b,r,a) in self.train_data and b<=a
            self.test_data= new_test
        elif (asymorsymorother==3):
            others=['_also_see', '_hypernym', '_synset_domain_topic_of']
            new_val=[(a,r,b) for (a,r,b) in self.test_data if r in others]#or (b,r,a) in self.train_data and b<=a
            self.test_data= new_val
        else:
            print("Testing with all relation types")


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
        return ['_also_see', '_derivationally_related_form', '_has_part', '_hypernym', '_instance_hypernym', '_member_meronym', '_member_of_domain_region', '_member_of_domain_usage', '_similar_to', '_synset_domain_topic_of', '_verb_group']
       # return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
