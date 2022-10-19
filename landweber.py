import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch_landweber(image, psf, num_iter=50, lam=0.7):
    """
    image: 4-dimensional input, NCHW format
    psf:   4-dimensional input, NCHW format
    """
    
    pad = psf.shape[-1]//2 + 1
    image = torch.nn.functional.pad(image, (pad,pad,pad,pad), mode='reflect')

    im_deconv = torch.full(image.shape, 0.5).to(device)

    for _ in range(num_iter):
        conv = torch.conv2d(im_deconv, psf, stride=1, padding=psf.shape[-1]//2)
        res = image - conv
        conv2 = torch.conv2d(res, torch.flip(psf, [0, 1, 2, 3]), stride=1, padding=psf.shape[-1]//2)
        im_deconv = im_deconv - lam * conv2
        im_deconv = torch.clip(im_deconv, -1, 1)
    
    return im_deconv[:,:,pad:-pad,pad:-pad]
  
