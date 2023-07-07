import sys, datetime
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import save_image

t = datetime.datetime.now()
savetime = t.strftime('%y%m%d')

class Logger(object):
    def __init__(self, modelname, datasetname, stream=sys.stdout):
        self.filename = f'{savetime}{modelname}4{datasetname}.log'
        self.terminal = stream
        self.log = open(self.filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def draw_splicingimgs(gener_tensor, real_tensor, recon_tensor, n_show, epoch, save_samples=False):

    if save_samples:
        save_image(gener_tensor.data[:25], "images/Generate_samples/%d.png" % epoch, nrow=10, normalize=True)
        save_image(real_tensor.data[:25], "images/Real_samples/%d.png" % epoch, nrow=10, normalize=True)
        save_image(recon_tensor.data[:25], "images/Recon_samples/%d.png" % epoch, nrow=10, normalize=True)

    gener_nparr = gener_tensor.reshape(n_show, 28, 28).cpu().numpy()
    recon_nparr = recon_tensor.reshape(n_show, 28, 28).cpu().numpy()
    real_nparr = real_tensor.reshape(n_show, 28, 28).cpu().numpy()

    fig, ax = plt.subplots(3, n_show, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.rcParams.update({'font.size': 20})
    fig.suptitle('Epoch {}'.format(epoch + 1))
    fig.text(0.04, 0.75, 'G(z, c)', ha='left')
    fig.text(0.04, 0.5, 'x', ha='left')
    fig.text(0.04, 0.25, 'G(E(x), c)', ha='left')

    for i in range(n_show):
        ax[0, i].imshow(gener_nparr[i], cmap='gray')
        ax[0, i].axis('off')
        ax[1, i].imshow(real_nparr[i], cmap='gray')
        ax[1, i].axis('off')
        ax[2, i].imshow(recon_nparr[i], cmap='gray')
        ax[2, i].axis('off')
    plt.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True