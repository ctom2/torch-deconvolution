from scipy.signal import convolve2d as conv2
from skimage import color, data
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn.functional as F

def torch_convolve(image, psf):
    p = psf.shape[-1]
    return F.conv2d(image.double(), psf.double(), padding=(p//2,p//2))

def torch_normalize(data):
    return (data - torch.min(data))/(torch.max(data) - torch.min(data))

# image values must be in range [0,1]
def torch_rl(image, psf, num_iter=15, clip=True, p=10):
    # padding to overcome ringing effect
    # image = torch.pad(image, ((padding,padding), (padding,padding)), 'mean')
    image = F.pad(image, (p,p,p,p), "constant", torch.mean(image))
    
    im_deconv = torch.full(image.shape, 0.5)

    # small regularization parameter used to avoid 0 divisions
    eps = 1e-6

    for _ in range(num_iter):
        conv = torch.clamp(torch_convolve(im_deconv, psf), eps, float("Inf"))
        relative_blur = image / conv
        term = torch_convolve(relative_blur, psf)
        term = torch.clamp(term, eps, float("Inf"))
        im_deconv = im_deconv * term
        
    return im_deconv[:,:,p:-p,p:-p]


def main():
    astro = color.rgb2gray(data.astronaut())
    psf = np.ones((5, 5)) / 25
    astro = conv2(astro, psf, 'same')

    astro = torch.from_numpy(astro).view(1,1,astro.shape[0],astro.shape[1]).double()
    psf = torch.from_numpy(psf).view(1,1,psf.shape[0],psf.shape[1]).double()

    deconvolved = torch_rl(astro, psf)

    plt.imshow(deconvolved.detach().numpy()[0,0], cmap='gray')
  
if __name__=="__main__":
    main()
