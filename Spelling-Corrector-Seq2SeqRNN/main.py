import argparse
from trainer import Trainer

parser = argparse.ArgumentParser(description='Spelling-Corrector')
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--rnn', default='lstm', type=str, help='gru/lstm')
parser.add_argument('--dim', default=512, type=int, help='hidden size of recurrent units')
parser.add_argument('--bidir', default=True, action='store_true')
parser.add_argument('--tf', default=1.0, type=float, help='teacher forcing ratio')
parser.add_argument('--word-drop', default=0.4, type=float, help='dropout technique')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optimizer', default='sgd', help='sgd/adam')
parser.add_argument('--checkname', default='test', type=str, help='saving checkname')

if __name__ == '__main__':
    args = parser.parse_args()
    model = Trainer(args)
    if args.eval:
        model.load_model()
        model.eval()
    else:
        model.train()