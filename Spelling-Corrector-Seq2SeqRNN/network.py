import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import random
from torch.nn.utils.rnn import pad_sequence

class Autoencoder(nn.Module):
    def __init__(self, args, EOS, dict_size=28):
        super(Autoencoder, self).__init__()
        self.args = args
        self.encoder = Encoder(args, dict_size)
        self.decoder = Decoder(args, EOS, dict_size)
    def forward(self, input, target, input_len, target_len):
        out, h = self.encoder(input, input_len)
        if self.training:
            if random.random() < self.args.tf:
                #print(target[0], target_len[0])
                out = self.decoder(target, target_len, h)
            else:
                out, out_seq = self.decoder.serial(target, h)
            return out
        else:
            out, out_seq = self.decoder.serial(target, h)
            return out, out_seq
            
        
class Encoder(nn.Module):
    def __init__(self, args, dict_size=28):
        super(Encoder, self).__init__()
        self.dim = args.dim
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim, padding_idx=0)
        self.rnntype = args.rnn
        if args.rnn == 'gru':
            self.rnn = nn.GRU(self.dim, self.dim,num_layers=1,batch_first=True,bidirectional=args.bidir)
        elif args.rnn == 'lstm':
            self.rnn = nn.LSTM(self.dim, self.dim,num_layers=1,batch_first=True,bidirectional=args.bidir)
        else:
            raise Exception('RNN cell implementation error, should be gru/lstm')
    def forward(self, words, lengths):
        
        words_embedding = self.embedding(words)
        #ordering
        sorted_len, indices = torch.sort(lengths, descending=True)
        _, reverse_sorting = torch.sort(indices)
        words_embedding = words_embedding[indices]
        lengths = lengths[indices]
        #lengths[lengths == 0] = 1
        packed_padded_sequence = pack_padded_sequence(words_embedding,lengths,batch_first=True)
        out, h = self.rnn(packed_padded_sequence)
        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        if self.rnntype == 'gru':
            h = h.permute(1, 0, 2).contiguous()
            h = h.view(-1, self.dim * (2 if self.bidir else 1))
            h = h[reverse_sorting]
        elif self.rnntype == 'lstm':
            hid, ceil = h
            hid = hid.permute(1, 0, 2).contiguous()
            hid = hid.view(-1, self.dim * (2 if self.bidir else 1))
            hid = hid[reverse_sorting]
            ceil = ceil.permute(1, 0, 2).contiguous()
            ceil = ceil.view(-1, self.dim * (2 if self.bidir else 1))
            ceil = ceil[reverse_sorting]
            h = (hid, ceil)
        out = out[reverse_sorting]

        return out, h

class Decoder(nn.Module):
    def __init__(self, args, EOS, dict_size=28):
        super(Decoder, self).__init__()
        self.dim = args.dim * (2 if args.bidir else 1)
        self.bidir = args.bidir
        self.embedding = nn.Embedding(dict_size, self.dim)
        self.rnntype = args.rnn
        self.tf = args.tf
        self.EOS = torch.tensor([[EOS]]).cuda()
        self.PAD = torch.tensor([[0]]).cuda()
        self.PAD_p = torch.zeros((1,dict_size)).cuda()
        self.PAD_p[0][0] = 1.0
        if args.rnn == 'gru':
            self.rnn = nn.GRU(self.dim, self.dim, num_layers=1, batch_first=True, bidirectional=False)
        elif args.rnn == 'lstm':
            self.rnn = nn.LSTM(self.dim, self.dim, num_layers=1, batch_first=True, bidirectional=False)
        else:
            raise Exception('RNN cell implementation error, should be gru/lstm')
        self.linear = nn.Linear(self.dim, dict_size)
    def serial(self, words, pre_h):
        
        
        word_sequence = []
        out_sequence = []
        for i in range(words.size(0)):
            maxlen = words.size(1)-1
            idx = 0
            token = words[i][0].unsqueeze(0).unsqueeze(1)
            if self.rnntype == 'gru':
                h = pre_h[i].unsqueeze(0).unsqueeze(1)
            elif self.rnntype == 'lstm':
                h = pre_h[0][i].unsqueeze(0).unsqueeze(1)
                ceil = torch.zeros((h.size(0),h.size(1),h.size(2))).cuda()
                h = (h,ceil)
            while True:
                #if random.random() < self.tf:
                if idx == 0:
                    token = words[i][0].unsqueeze(0).unsqueeze(1)
                elif self.training and random.random() < self.tf:
                    token = words[i][idx].unsqueeze(0).unsqueeze(1)
                token = token.detach()
                word_embedding = self.embedding(token)
                out, h = self.rnn(word_embedding, h)
                out = self.linear(out)
                _, text = out.topk(1)
                token = text.squeeze(0)
                out = out.squeeze(0)
                if idx == 0:
                    word_seq = token
                    out_seq = out
                else:
                    word_seq = torch.cat((word_seq, token),dim=0)
                    out_seq = torch.cat((out_seq, out),dim=0)
                idx += 1
                if idx == maxlen or token.item() == self.EOS:
                    break
            while idx < maxlen:
                word_seq = torch.cat((word_seq, self.PAD),dim=0)
                out_seq = torch.cat((out_seq, self.PAD_p),dim=0)
                idx += 1
            word_sequence.append(word_seq)
            out_sequence.append(out_seq)
        word_sequence = pad_sequence(word_sequence, batch_first=True)
        out_sequence = pad_sequence(out_sequence, batch_first=True)
        return out_sequence, word_sequence
    def forward(self, words, lengths, pre_h):
        words_embedding = self.embedding(words)
        #ordering
        sorted_len, indices = torch.sort(lengths, descending=True)
        _, reverse_sorting = torch.sort(indices)
        words_embedding = words_embedding[indices]
        lengths = lengths[indices]
        if self.rnntype == 'gru':
            pre_h = pre_h[indices]
            pre_h = pre_h.unsqueeze(0)
        elif self.rnntype == 'lstm':
            hid, _ = pre_h
            hid = hid[indices]
            hid = hid.unsqueeze(0)
            ceil = torch.zeros((hid.size(0),hid.size(1),hid.size(2))).cuda()
            pre_h = (hid, ceil)
        packed_padded_sequence = pack_padded_sequence(words_embedding,lengths,batch_first=True)
        out, h = self.rnn(packed_padded_sequence, pre_h)
        out, out_len = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        #h = h.permute(1, 0, 2).contiguous()
        #h = h.view(-1, self.dim)
        #h = h[reverse_sorting]
        out = out[reverse_sorting]
        out = self.linear(out)
        return out