from trainer import Gans_trainer
import argparse

parser = argparse.ArgumentParser(description="Let's play GANs")
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--checkname', default='best', type=str, help='saving checkname')
parser.add_argument('--datapath', default='./', type=str, help='training/testing datapath')
parser.add_argument('--ncritic', default=1, type=int)
parser.add_argument('--projectd', default=False, action='store_true')
parser.add_argument('--cls-param',default=2, type=float, help='parameter of classification loss')
parser.add_argument('--condition-mode', default='embedding', type=str, help='condition type (embedding or onehot)')
parser.add_argument('--testname', default='test', type=str, help='')
parser.add_argument('--loadepoch', default=91, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    model = Gans_trainer(args)
    if args.eval:
        model.load_from_checkpoint(epoch=args.loadepoch) #best accuracy
        model.eval()
    else:
        model.train()

