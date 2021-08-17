import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

def imshow(img):
    img = img * 0.5 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()
    
def save_fig(img, dir, name='test'):
    img = img * 0.5 + 0.5
    iter = img.size(0) // 16
    #for i in range(iter):
    #    save_image(img[i*16:(i+1)*16,:].cpu(), dir+'/'+name+str(i)+'.png')
    out = make_grid(img)
    save_image(out, dir+'/'+name+'.png')
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)