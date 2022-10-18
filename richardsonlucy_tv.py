import torch

# Richardson-Lucy with total variation (TV) regularisation
def torch_richardson_lucy_tv(image, psf, num_iter=10, lam=2e-2):
    """
    image: 4-dimensional input, NCHW format
    psf:   4-dimensional input, NCHW format
    """
    
    im_deconv = torch.full(image.shape, 0.5)
    psf_mirror = torch.flip(psf, (-2,-1))

    eps = 1e-12
    reg = 1

    for _ in range(num_iter):
        # tv
        if lam > 0:
            grad_torch = torch.gradient(im_deconv[0,0], axis=(0, 1))
            norm_torch = torch.sqrt(torch.square(grad_torch[0])+torch.square(grad_torch[1])) + eps
            grad_torch = torch.stack(grad_torch)/norm_torch
            div_torch = torch.gradient(grad_torch[0], axis=0)[0] + torch.gradient(grad_torch[1], axis=1)[0]
            reg = 1/(1-div_torch*lam)


        conv = torch.conv2d(im_deconv, psf, stride=1, padding='same') + eps
        relative_blur = image / conv
        im_deconv *= (torch.conv2d(relative_blur, psf_mirror, stride=1, padding='same') + eps) * reg
        im_deconv = torch.clip(im_deconv, -1, 1)
    
    return im_deconv
