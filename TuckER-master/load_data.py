import pickle
import random
import math
from collections import defaultdict

class Data:
    # data_opt = 0: no restriction // 1: only sym // 2: only asym // 3: both sym and asym // 4: all except sym and asym
    def __init__(self, data_dir="data/WN18RR/", reverse = True, subset_percentage=1.0, data_opt=0):
        
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = sorted(list(set(self.train_relations + self.valid_relations + self.test_relations)))
        
        if data_opt:
            self.restrict_data(data_opt)
        else:
            print("Training with all relation types")
            if subset_percentage!=1.0: self.get_percentage_train_data(subset_percentage)
            else: print('Training with full training set')

        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        

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
    
    
    def get_percentage_train_data(self, percentage):
        print('Training with {}\% of the training set'.format(percentage*100))
        rel_dic = defaultdict(list)
        for triple in self.train_data:
            rel_dic[(triple[1])].append(triple)
        for key in redic:
            rel_dic[key] = random.sample(rel_dic[key], math.floor(len(rel_dic[key])*percentage))
        data = []
        for key in redic:
            data = data + redic[key]
        self.train_data = data
    
    
    def restrict_data(self, data_opt):
        if (data_opt == 1):
            sym = ['_derivationally_related_form', '_similar_to', '_verb_group']
            print('Dataset restricted to symmetric relations :{}'.format(sym))
            
            temp_data = [(a,r,b) for (a,r,b) in self.train_data if r in sym]
            self.train_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.valid_data if r in sym]
            self.valid_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.test_data if r in sym]
            self.test_data = temp_data
            
        elif (data_opt == 2):
            asym = ['_has_part', '_instance_hypernym', '_member_meronym','_member_of_domain_region','_member_of_domain_usage']
            print('Dataset restricted to asymmetric relations :{}'.format(asym))
            
            temp_data = [(a,r,b) for (a,r,b) in self.train_data if r in asym]
            self.train_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.valid_data if r in asym]
            self.valid_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.test_data if r in asym]
            self.test_data = temp_data
            
        elif (data_opt == 3):
            sanda=['_derivationally_related_form', '_similar_to', '_verb_group','_has_part',
                   '_instance_hypernym', '_member_meronym','_member_of_domain_region','_member_of_domain_usage']
            print('Dataset restricted to symmetric and asymmetric relations :{}'.format(sanda))
            
            temp_data = [(a,r,b) for (a,r,b) in self.train_data if r in sanda]
            self.train_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.valid_data if r in sanda]
            self.valid_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.test_data if r in sanda]
            self.test_data = temp_data
            
        elif (data_opt == 4):
            others = ['_also_see', '_hypernym', '_synset_domain_topic_of']
            print('Dataset restricted to all relations except symmetric and asymmetric:{}'.format(others))
            
            temp_data = [(a,r,b) for (a,r,b) in self.train_data if r in others]
            self.train_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.valid_data if r in others]
            self.valid_data = temp_data
            temp_data = [(a,r,b) for (a,r,b) in self.test_data if r in others]
            self.test_data = temp_data