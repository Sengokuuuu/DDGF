import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
class_num = 951

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def structure_tensor(x, sigma=1.0):
    if x.dim() == 4 and x.size(1) > 1:
        x = torch.mean(x, dim=1, keepdim=True)
    Ix = torch.gradient(x, dim=3)[0]
    Iy = torch.gradient(x, dim=2)[0]
    Jxx = Ix ** 2
    Jxy = Ix * Iy
    Jyy = Iy ** 2
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    kernel = kernel.view(1, 1, 3, 3).to(x.device)
    Jxx = torch.nn.functional.conv2d(Jxx, kernel, padding=1)
    Jxy = torch.nn.functional.conv2d(Jxy, kernel, padding=1)
    Jyy = torch.nn.functional.conv2d(Jyy, kernel, padding=1)
    ST = torch.stack([Jxx, Jxy, Jxy, Jyy], dim=1).view(x.size(0), 2, 2, x.size(2), x.size(3))
    return ST

def fractional_gradient(x, *, alpha_prior):
    x = torch.mean(x, dim=1, keepdim=True) if x.size(1) > 1 else x
    freq_x = torch.fft.fftfreq(x.size(-1), device=x.device).view(1, 1, 1, -1)
    freq_y = torch.fft.fftfreq(x.size(-2), device=x.device).view(1, 1, -1, 1)
    fft_x = torch.fft.fft2(x)
    grad_x = torch.fft.ifft2((1j * 2 * np.pi * freq_x) ** alpha_prior * fft_x).real
    grad_y = torch.fft.ifft2((1j * 2 * np.pi * freq_y) ** alpha_prior * fft_x).real
    return torch.stack([grad_x, grad_y], dim=1) # [B, 2, 1, H, W]

def gradient_regularization_prior(x, *, alpha_prior, lambda_prior):
    x.requires_grad_(True)
    ST = structure_tensor(x) # [B, 2, 2, H, W]
    grad_alpha = fractional_gradient(x, alpha_prior=alpha_prior) # [B, 2, 1, H, W]
    grad_alpha = grad_alpha.squeeze(2) # [B, 2, H, W]
    weighted_grad = torch.einsum('bikhw,bkhw->bihw', ST, grad_alpha)
    R = torch.sum(torch.abs(weighted_grad)) # L1 norm
    grad_R = torch.autograd.grad(R, x)[0]
    return lambda_prior * grad_R

def gaussian_lowpass_filter(freq_dist, *, omega_prior, high_pass=False):
    sigma = 1.0 / (2.0 * np.pi * omega_prior)
    gaussian_mask = torch.exp(-0.5 * (freq_dist ** 2) / (sigma ** 2))
    if high_pass:
        gaussian_mask = 1.0 - gaussian_mask
    return gaussian_mask

def frequency_domain_prior(x, *, omega_prior):
    x_gray = torch.mean(x, dim=1, keepdim=True) if x.size(1) > 1 else x
    fft_x = torch.fft.fft2(x_gray)
    freq_h = torch.fft.fftfreq(x.size(-2), device=x.device).view(1, 1, -1, 1)
    freq_w = torch.fft.fftfreq(x.size(-1), device=x.device).view(1, 1, 1, -1)
    freq_dist = torch.sqrt(freq_h ** 2 + freq_w ** 2)
    M = gaussian_lowpass_filter(freq_dist, omega_prior=omega_prior, high_pass=False) # 默认低通
    modulated_fft = fft_x * M
    modulated_x = torch.fft.ifft2(modulated_fft).real
    return modulated_x.repeat(1, x.size(1), 1, 1)

def DDGF_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None, *, lambda_prior, alpha_prior, omega_prior):
    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]
        times = get_schedule_jump(
            config.time_travel.T_sampling,
            config.time_travel.travel_length,
            config.time_travel.travel_repeat,
        )
        time_pairs = list(zip(times[:-1], times[1:]))
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0:
                j = -1
            if j < i: # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=xt.device) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
                if et.size(1) == 6:
                    et = et[:, :3]
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())
                with torch.enable_grad():
                    grad_prior = gradient_regularization_prior(
                        x0_t_hat, alpha_prior=alpha_prior, lambda_prior=lambda_prior
                    )
                x0_t_hat = x0_t_hat - grad_prior.detach()
                # x0_t_hat = frequency_domain_prior(x0_t_hat, omega_prior=omega_prior)
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                xs.append(xt_next.to('cpu'))
    return [xs[-1]], [x0_preds[-1]]

def DDGF_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, *, lambda_prior, alpha_prior, omega_prior):
    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]
        times = get_schedule_jump(
            config.time_travel.T_sampling,
            config.time_travel.travel_length,
            config.time_travel.travel_repeat,
        )
        time_pairs = list(zip(times[:-1], times[1:]))
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0:
                j = -1
            if j < i: # normal sampling
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn is None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=xt.device) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
                if et.size(1) == 6:
                    et = et[:, :3]
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
                x0_t_hat = x0_t - A_funcs.Lambda(
                    A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(x0_t.size(0), -1),
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta
                ).reshape(*x0_t.size())
                with torch.enable_grad():
                    grad_prior = gradient_regularization_prior(
                        x0_t_hat, alpha_prior=alpha_prior, lambda_prior=lambda_prior
                    )
                x0_t_hat = x0_t_hat + grad_prior.detach()
                x0_t_hat = frequency_domain_prior(x0_t_hat, omega_prior=omega_prior)
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1),
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)
                ).reshape(*x0_t.size())
                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                xs.append(xt_next.to('cpu'))
    return [xs[-1]], [x0_preds[-1]]

def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1
    t = T_sampling
    ts = []
    while t >= 1:
        t = t - 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)
    ts.append(-1)
    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])
    assert times[-1] == -1, times[-1]
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)