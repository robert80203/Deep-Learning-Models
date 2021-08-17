from dataset import iclevr_dataset, test_dataset
from utils import imshow, save_fig, weights_init
import torch
import torch.nn.functional as F
from network import *
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from tqdm import tqdm
import os
#from lab6_packages.evaluator import evaluation_model

torch.manual_seed(100)

class Gans_trainer():
    def __init__(self, args):
        self.args = args
        dataset = iclevr_dataset(args.datapath)
        test_set = test_dataset(args.testname)
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=False, num_workers=1)
        self.imgsize = 64
        self.best = 0.0
        self.BCE = nn.BCELoss()
        self.CE = nn.CrossEntropyLoss()
        self.d_net = D(self.args.projectd).cuda()
        self.g_net = G(self.args.condition_mode).cuda()
        self.d_net.apply(weights_init)
        self.g_net.apply(weights_init)
        self.D_opt = optim.Adam(self.d_net.parameters(), lr=args.lr, betas=(0.5, 0.99))
        self.G_opt = optim.Adam(self.g_net.parameters(), lr=args.lr*4, betas=(0.5, 0.99))
        self.batch_len = len(self.train_loader)
        self.infinite_loader = self.get_infinite_batches(self.train_loader)
        #self.eval_model = evaluation_model()
        if not os.path.isdir('logs/'+args.checkname):
            os.mkdir('logs/'+args.checkname)
    def gradient_penalty(self, discriminator_out, data_point):
        batch_size = data_point.size(0)
        grad_dout = autograd.grad(outputs=discriminator_out.sum(),
                              inputs=data_point,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]

        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == data_point.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg
    def get_infinite_batches(self, data_loader):
        while True:
            for step, data in enumerate(data_loader):
                yield data
    def compute_D_loss(self, fout, rout, faux, raux, x_, labels):
        #Fake loss
        D_fake_loss = fout.mean()#F.relu(1. + fout).mean()
        fake_aux = self.BCE(faux, labels)
        #Real loss
        D_real_loss = - rout.mean()#F.relu(1. - rout).mean()
        real_aux = self.BCE(raux, labels)
        #real_aux = self.CE(raux, labels)
        gp = self.gradient_penalty(rout, x_).mean() * 10
        aux = (fake_aux + real_aux) * self.args.cls_param/2
        logit = D_fake_loss + D_real_loss + gp
        D_loss = (logit + aux) / 2.0
        return D_loss, logit, fake_aux, real_aux
    def compute_G_loss(self, gout, gaux, labels):
        logit = -gout.mean()
        aux = self.BCE(gaux, labels) * self.args.cls_param
        G_loss = logit + aux
        return G_loss, logit, aux
    def save_to_checkpoint(self, epoch):
        print("==========> Save checkpoint...")
        torch.save({
            'epoch' : epoch,
            'model_g' : self.g_net.state_dict(),
            'optimizer_g' : self.G_opt.state_dict(),
            'model_d' : self.D_opt.state_dict(),
            'optimizer_d' : self.d_net.state_dict(),
        }, 'logs/' + self.args.checkname + '/' + str(epoch)+"_checkpoint.pth")
    def load_from_checkpoint(self, epoch=5):
        print('==========> Load from %s ...'%(self.args.checkname))
        load_dir = './logs/'+self.args.checkname+'/'+str(epoch)+'_checkpoint.pth'
        checkpoint = torch.load(load_dir)
        self.g_net.load_state_dict(checkpoint['model_g'])
    def eval(self):
        self.g_net.eval()
        with torch.no_grad():
            for iter, data in enumerate(self.test_loader):
                onehot_labels, labels = data
                onehot_labels = onehot_labels.cuda()
                labels = labels.long().cuda()
                z = torch.randn(labels.size(0), 64).cuda()
                if self.args.condition_mode == 'embedding':
                    fake_image = self.g_net(z, labels)
                else:
                    fake_image = self.g_net(z, onehot_labels)
                save_fig(fake_image.cpu(), 'logs/'+self.args.checkname, self.args.testname)
                
                # recall, precision, f1_score = self.eval_model.eval(fake_image, onehot_labels)
                # print('recall:%.3f, precision:%.3f, f1_score:%.3f'%(recall, precision, f1_score))
        self.g_net.train()
        #return f1_score
    def train(self):
        for epoch in range(self.args.epoch):
            pbar = tqdm(self.train_loader, ncols=100)
            
            step = 0
            gloss_total = 0.0
            dloss_total = 0.0
            for data in pbar:
                images, onehot_labels, labels = data
                images = F.interpolate(images, size=(self.imgsize,self.imgsize), mode='bilinear', align_corners=True).cuda()
                labels = labels.long().cuda()
                onehot_labels = onehot_labels.cuda()
                z = torch.randn(images.size(0), 64).cuda()
                #latent = torch.cat((z, labels),dim=1)
                if self.args.condition_mode == 'embedding':
                    fake_image = self.g_net(z, labels)
                else:
                    fake_image = self.g_net(z, onehot_labels)
                #########################
                #Training discriminator
                #########################
                self.d_net.zero_grad()
                images.requires_grad = True
                real_logits, real_aux = self.d_net(images, labels)
                fake_logits, fake_aux = self.d_net(fake_image.detach(), labels)
                D_loss, dlogit, faux_loss, raux_loss = self.compute_D_loss(fake_logits, real_logits, fake_aux, real_aux, images, onehot_labels)
                #D_loss, dlogit, faux_loss, raux_loss = self.compute_D_loss(fake_logits, real_logits, fake_aux, real_aux, images, labels)
                D_loss.backward()
                self.D_opt.step()
                
                #########################
                #Training generator
                #########################
                images.requires_grad = False
                self.g_net.zero_grad()
                g_logits, g_aux = self.d_net(fake_image, labels)
                G_loss, glogit, gaux_loss = self.compute_G_loss(g_logits, g_aux, onehot_labels)
                #G_loss, glogit, gaux_loss = self.compute_G_loss(g_logits, g_aux, labels)
                
                G_loss.backward()
                self.G_opt.step()
                

                step += 1
                pbar.set_description('epoch'+str(epoch))
                pbar.set_postfix({'Gloss':(glogit.item()),
                                  'Dloss':(dlogit.item()),
                                  'GCls':(gaux_loss.item()),
                                  'DCls(f)':(faux_loss.item()),
                                  'DCls(r)':(raux_loss.item())},refresh=True)
            self.save_to_checkpoint(1000)
                # if step % (self.batch_len//4) == 0:
                #     acc = self.eval()
                #     if acc >= self.best:
                #         print('Save best model...')
                #         self.best = acc
                #         self.save_to_checkpoint(1000)