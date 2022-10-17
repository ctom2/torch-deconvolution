import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Richardson-Lucy with total variation (TV) regularisation
def torch_richardson_lucy_tv(image, psf, num_iter=50, lam=2e-2):
    """
    image: 4-dimensional input, NCHW format
    psf:   4-dimensional input, NCHW format
    """
    
    pad = psf.shape[-1]//2 + 1
    image = torch.nn.functional.pad(image, (pad,pad,pad,pad), mode='reflect')

    im_deconv = torch.full(image.shape, 0.5).to(device)
    psf_mirror = torch.transpose(psf, -2, -1)

    eps = 1e-12
    reg = 0

    for _ in range(num_iter):
        # tv
        if lam > 0:
            grad_torch = torch.gradient(im_deconv[0,0], axis=(0, 1))
            norm_torch = torch.sqrt(torch.square(grad_torch[0])+torch.square(grad_torch[1])) + eps
            grad_torch = torch.stack(grad_torch)/norm_torch
            div_torch = torch.gradient(grad_torch[0], axis=0)[0] + torch.gradient(grad_torch[1], axis=1)[0]
            reg = div_torch*lam 

        conv = torch.conv2d(im_deconv, psf, stride=1, padding=psf.shape[-1]//2) + eps
        relative_blur = image / conv
        im_deconv *= (torch.conv2d(relative_blur, psf_mirror, stride=1, padding=psf.shape[-1]//2) + eps) + reg
        im_deconv = torch.clip(im_deconv, -1, 1)
    
    return im_deconv[:,:,pad:-pad,pad:-pad]