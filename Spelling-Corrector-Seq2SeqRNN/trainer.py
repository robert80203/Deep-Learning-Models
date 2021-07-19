import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import json
from network import Autoencoder
import torch.optim as optim
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

class spelldataset(Dataset):
    def __init__(self, data, chartoidx, idxtochar,transform=None):
        input_tensor = []
        target_tensor = []
        input_len_tensor = []
        target_len_tensor = []
        for i in data:
            for word in i['input']:
                input_len_tensor.append(len(word))
                target_len_tensor.append(len(i['target'])+2)
                input = []
                for character in word:
                    input.append(chartoidx[character])
                input_tensor.append(torch.tensor(input).long())
                target = [27]
                for character in i['target']:
                    target.append(chartoidx[character])
                target.append(28)
                target.append(0)#prevent too long prediction
                target_tensor.append(torch.tensor(target).long())
        data_size = len(input_tensor)
        output_tensor = pad_sequence(input_tensor + target_tensor, batch_first=True)
        self.target_ = output_tensor[data_size:]
        self.input_ = output_tensor[:data_size]
        self.input_len_ = torch.tensor(input_len_tensor).long()
        self.target_len_ = torch.tensor(target_len_tensor).long()
    
    def __len__(self):
        return len(self.input_)
    
    def __getitem__(self, idx):
        return self.input_[idx], self.target_[idx], self.input_len_[idx], self.target_len_[idx]

class Trainer():
    def __init__(self, args):
        self.args = args
        self.chartoidx = {
            'PAD':0,
            'a':1,'b':2,'c':3,'d':4,'e':5,
            'f':6,'g':7,'h':8,'i':9,'j':10,
            'k':11,'l':12,'m':13,'n':14,'o':15,
            'p':16,'q':17,'r':18,'s':19,'t':20,
            'u':21,'v':22,'w':23,'x':24,'y':25,
            'z':26,'SOS':27,'EOS':28,'UNK':29
        }
        self.idxtochar = {
            0:'PAD',
            1:'a',2:'b',3:'c',4:'d',5:'e',
            6:'f',7:'g',8:'h',9:'i',10:'j',
            11:'k',12:'l',13:'m',14:'n',15:'o',
            16:'p',17:'q',18:'r',19:'s',20:'t',
            21:'u',22:'v',23:'w',24:'x',25:'y',
            26:'z',27:'SOS',28:'EOS',29:'UNK'
        }
        self.EOS = self.chartoidx['EOS']
        self.SOS = self.chartoidx['SOS']
        self.UNK = self.chartoidx['UNK']
        self.dict_size = len(self.chartoidx)
        self.model = Autoencoder(args, self.EOS, self.dict_size).cuda()
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
            self.scheduler = None
        test = self.load_data('test')
        dataset_test = spelldataset(test, self.chartoidx, self.idxtochar)
        self.test_loader = DataLoader(dataset_test, batch_size=50, shuffle=False, num_workers=1)
        data = self.load_data('train')
        dataset_train = spelldataset(data, self.chartoidx, self.idxtochar)
        self.train_loader = DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=1)
    
    def word_dropout(self, input):
        input_decode = input.clone()
        unk_pos = torch.zeros_like((input_decode)).float() + self.args.word_drop
        unk_pos = torch.bernoulli(unk_pos)
        input_decode[unk_pos == 1] = self.chartoidx['UNK']
        return input_decode
    
    def load_data(self, mode='train'):
        if mode == 'train':
            with open('./data/train.json','r') as fp:
                data = json.load(fp)
        else:
            with open('./data/test.json','r') as fp:
                data = json.load(fp)
        return data
    
    def criterion(self, out, target, loss_func):  
        target_ = target[:,1:out.size()[1]+1]
        target_ = target_.contiguous().view(-1)
        _, idx = out.topk(1)
        #print(idx, idx.size())
        out = out.view(-1, self.dict_size)
        loss = loss_func(out,target_)
        return loss
    
    def compute_bleu(self, reference, output):
        cc = SmoothingFunction()
        if len(reference) == 3:
            w = (0.33,0.33,0.33)
        else:
            w = (0.25,0.25,0.25,0.25)
        return sentence_bleu([reference], output, weights=w,smoothing_function=cc.method1)
    
    def show_results(self, input, target, pred):
        idx = 0
        score = 0.0
        for text in pred:
            pred_text = self.show_text(text)
            input_text = self.show_text(input[idx])
            target_text = self.show_text(target[idx])
            print('===========================')
            print('input:\t%s'%(input_text))
            print('target:\t%s'%(target_text))
            print('pred:\t%s'%(pred_text))
            idx += 1
            score += self.compute_bleu(target_text, pred_text)
        score /= idx
        print('BLEU-4 score:%.4f'%(score))
    
    def show_text(self, x):
        string = ''
        for i in x:
            if i.item() == self.SOS or i.item() == self.EOS:
                continue
            elif i.item() == 0:
                break
            string += self.idxtochar[i.item()]
        return string
    
    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, 'weights/'+self.args.checkname+str(epoch)+'.pth')
    
    def load_model(self):
        checkpoint = torch.load('weights/'+self.args.checkname+'.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def train(self):
        cri = nn.CrossEntropyLoss(ignore_index=self.chartoidx['PAD'])
        for epoch in range(self.args.epoch):
            pbar = tqdm(self.train_loader)
            for data in pbar:
                self.model.zero_grad()
                input, target, input_len, target_len = data
                input, target, input_len, target_len = input.cuda(), target.cuda(), input_len.cuda(), target_len.cuda()
                d_target = self.word_dropout(target)
                out = self.model(input, d_target, input_len, target_len)
                loss = self.criterion(out, target, cri)
                loss.backward()
                self.optimizer.step()
                pbar.set_description('epoch '+str(epoch))
                pbar.set_postfix({'CE Loss': (loss.item()),}, refresh=True)
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % 5 == 0:
                #self.show_results(input, target, out)
                self.save_model(epoch)
                self.eval()
    
    def eval(self):
        print('Evaluating...')
        self.model.eval()
        for iter, data in enumerate(self.test_loader):
            input, target, input_len, target_len = data
            input, target, input_len, target_len = input.cuda(), target.cuda(), input_len.cuda(), target_len.cuda()
            _, out = self.model(input, target, input_len, target_len)
            self.show_results(input, target, out)
        self.model.train()