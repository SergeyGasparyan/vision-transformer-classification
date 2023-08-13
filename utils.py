import torch
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
