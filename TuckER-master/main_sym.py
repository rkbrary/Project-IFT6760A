import wandb
wandb.login('1e505430989c86455d2d70e1ef990b4bc50cb69c')
wandb.init(project="ift6760-exp",anonymous='allow')
wandb.save("*.pt")
from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
# from model import *
from model_sym import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse


class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0., bk=False):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2, "bk": bk}
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    
    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0]).type(torch.LongTensor)
            r_idx = torch.tensor(data_batch[:,1]).type(torch.LongTensor)
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        wandb.log({
            'Hits @10':np.mean(hits[9]),
            'Hits @3':np.mean(hits[2]),
            'Hits @1':np.mean(hits[0]),
            'MRR':np.mean(1./np.array(ranks))})
        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))


    def retrain(self, num_it, model_state_path='model_state.pt'):
        checkpoint = torch.load(model_state_path)

        print("Resuming training at epoch {} for {} epochs".format(checkpoint['epoch'], num_it))
        if self.kwargs["bk"]:
            print("Background knowledge=True")
            print("Model=",args.dataset)
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        wandb.watch(model, log=None)
        if self.cuda:
            model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, num_it+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0]).type(torch.LongTensor)
                r_idx = torch.tensor(data_batch[:,1]).type(torch.LongTensor)
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print('Iteration:'+str(it+checkpoint['epoch']))
            print('Training Time:'+str(time.time()-start_train))    
            print('Loss:'+str(np.mean(losses)))
            torch.save({
                'epoch': (it+checkpoint['epoch']),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }, model_state_path)
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data)
        with torch.no_grad():
            print("Test:")
            start_test = time.time()
            self.evaluate(model, d.test_data)
            print("Testing time:"+str(time.time()-start_test))

    def train_and_eval(self, path='model_state.pt'):
        print("Training the TuckER model...")
        if self.kwargs["bk"]:
            print("Background knowledge=True")
            print("Model=",args.dataset)
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        wandb.watch(model, log=None)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0]).type(torch.LongTensor)
                r_idx = torch.tensor(data_batch[:,1]).type(torch.LongTensor)
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print('Iteration:'+str(it))
            print('Training Time:'+str(time.time()-start_train))    
            print('Loss:'+str(np.mean(losses)))
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }, path)
            model.eval()
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data)
        with torch.no_grad():
            print("Test:")
            start_test = time.time()
            self.evaluate(model, d.test_data)
            print("Testing time:"+str(time.time()-start_test))
           

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WN18RR", nargs="?",
                    help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=50, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=30, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.2, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.2, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.3, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.2, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--bk", type=bool, default=False, nargs="?",
                    help="Whether to use background knowledge or not.")

    args = parser.parse_args()
    # Saving the configuration
    config = wandb.config
    config.lr = args.lr
    config.dr = args.dr
    config.edim = args.edim
    config.rdim = args.rdim
    config.input_dropout = args.input_dropout
    config.hidden_dropout1 = args.hidden_dropout1
    config.hidden_dropout2 = args.hidden_dropout2
    config.label_smoothing = args.label_smoothing
    config.batch_size = args.batch_size
    config.num_iterations = args.num_iterations
    config.bk=args.bk
    
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=False)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing,
                            bk=args.bk)
    path='model_state.pt'
    if args.bk: path='model_state_sym.pt'
    experiment.train_and_eval(path)
                

