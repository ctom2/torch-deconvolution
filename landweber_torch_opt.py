import torch

def landweber_torch_opt(y, k, num_iter=50, lam=0.7):
    xi = torch.nn.Parameter(torch.ones(y.shape)*0.5).float()

    optimizer = torch.optim.SGD([xi], lr=lam)

    for it in range(num_iter):
        optimizer.zero_grad()

        y_out = torch.conv2d(xi, k, stride=1, padding='same')
        loss = torch.sum(torch.pow((y_out - y),2))/2

        loss.backward()
        optimizer.step()

    return y_out
