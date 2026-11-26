"""ChequeGen core: noise schedule, conditional U-Net, DDPM / DDIM / DPM-Solver++ samplers, SPECIMEN cheque renderer.
Generated from ChequeGen.ipynb so the app and the notebook share one implementation."""
import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {"sample_batch": 500}

class Schedule:
    # Linear beta schedule and the quantities derived from it
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def q_sample(self, x0, t, noise):
        # forward process: jump straight from x0 to x_t
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class ResBlock(nn.Module):
    # GroupNorm -> SiLU -> Conv, with the time+label embedding added in the middle
    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in);  self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out); self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class CondUNet(nn.Module):
    # Class-conditional U-Net for 28x28 digits. Label 10 = 'no label' (classifier-free guidance)
    def __init__(self, ch=(32, 64, 128), emb_dim=128, n_classes=10):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(n_classes + 1, emb_dim)
        self.inc = nn.Conv2d(1, ch[0], 3, padding=1)
        self.down1 = ResBlock(ch[0], ch[0], emb_dim)            # 28x28
        self.down2 = ResBlock(ch[0], ch[1], emb_dim)            # 14x14
        self.down3 = ResBlock(ch[1], ch[2], emb_dim)            # 7x7
        self.mid = ResBlock(ch[2], ch[2], emb_dim)
        self.up3 = ResBlock(ch[2] * 2, ch[1], emb_dim)          # 7x7
        self.up2 = ResBlock(ch[1] * 2, ch[0], emb_dim)          # 14x14
        self.up1 = ResBlock(ch[0] * 2, ch[0], emb_dim)          # 28x28
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 1, 3, padding=1))
    def forward(self, x, t, y):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.label_emb(y)
        d1 = self.down1(self.inc(x), emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), emb)
        d3 = self.down3(F.avg_pool2d(d2, 2), emb)
        m = self.mid(d3, emb)
        u3 = self.up3(torch.cat([m, d3], 1), emb)
        u2 = self.up2(torch.cat([F.interpolate(u3, scale_factor=2), d2], 1), emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2), d1], 1), emb)
        return self.out(u1)

@torch.no_grad()
def guided_eps(net, x, t, y, w):
    # one batched pass for conditional + unconditional predictions (fp16 on GPU)
    tt = torch.full((2 * len(x),), t, device=x.device, dtype=torch.long)
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.is_cuda):
        e = net(torch.cat([x, x]), tt, torch.cat([y, torch.full_like(y, 10)]))
    e_c, e_u = e.float().chunk(2)
    return (1 + w) * e_c - w * e_u

@torch.no_grad()
def predict_x0(sched, x, t, eps):
    ab = sched.alpha_bar[t]
    x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
    eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()            # re-derive eps consistent with the clipped x0
    return x0, eps

@torch.no_grad()
def ddpm_sample(net, sched, y, w=1.5, generator=None):
    # Baseline: ancestral DDPM sampling over all T steps
    x = torch.randn((len(y), 1, 28, 28), device=y.device, generator=generator)
    for t in reversed(range(sched.T)):
        x0, _ = predict_x0(sched, x, t, guided_eps(net, x, t, y, w))
        ab_t = sched.alpha_bar[t]
        ab_prev = sched.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=x.device)
        mean = (ab_prev.sqrt() * sched.betas[t] / (1 - ab_t)) * x0 + (sched.alphas[t].sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
        if t > 0:
            var = (1 - ab_prev) / (1 - ab_t) * sched.betas[t]
            x = mean + var.sqrt() * torch.randn(x.shape, device=x.device, generator=generator)
        else:
            x = mean
    return x

@torch.no_grad()
def ddim_sample(net, sched, y, steps=50, w=1.5, eta=0.0, generator=None):
    # Improvement 1: DDIM on a sub-sequence of `steps` timesteps (eta=0 -> deterministic)
    taus = np.unique(np.linspace(0, sched.T - 1, steps).round().astype(int))[::-1]
    x = torch.randn((len(y), 1, 28, 28), device=y.device, generator=generator)
    for i, t in enumerate(taus):
        x0, eps = predict_x0(sched, x, int(t), guided_eps(net, x, int(t), y, w))
        ab_t = sched.alpha_bar[t]
        ab_prev = sched.alpha_bar[taus[i + 1]] if i + 1 < len(taus) else torch.tensor(1.0, device=x.device)
        sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if eta > 0 and i + 1 < len(taus):
            x = x + sigma * torch.randn(x.shape, device=x.device, generator=generator)
    return x

@torch.no_grad()
def dpm_solver_pp_sample(net, sched, y, steps=20, w=1.5, generator=None):
    # Improvement 2: DPM-Solver++(2M) — multistep 2nd-order solver in data (x0) prediction form
    ts = np.linspace(sched.T - 1, 0, steps + 1).round().astype(int)
    alpha = lambda t: sched.alpha_bar[t].sqrt()
    sigma = lambda t: (1 - sched.alpha_bar[t]).sqrt()
    lam = lambda t: torch.log(alpha(t) / sigma(t))
    x = torch.randn((len(y), 1, 28, 28), device=y.device, generator=generator)
    x0_prev, h_prev = None, None
    for i in range(steps):
        s, t = int(ts[i]), int(ts[i + 1])
        x0, _ = predict_x0(sched, x, s, guided_eps(net, x, s, y, w))
        h = lam(t) - lam(s)
        if x0_prev is None or i == steps - 1:        # 1st-order start and final step (stable at few steps)
            D = x0
        else:
            r = h_prev / h
            D = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * x0_prev
        x = (sigma(t) / sigma(s)) * x - alpha(t) * torch.expm1(-h) * D
        x0_prev, h_prev = x0, h
    return x

def sample(kind, net, sched, y, steps, w, generator=None):
    if kind == 'ddpm': return ddpm_sample(net, sched, y, w, generator=generator)
    if kind == 'ddim': return ddim_sample(net, sched, y, steps, w, generator=generator)
    if kind == 'dpm':  return dpm_solver_pp_sample(net, sched, y, steps, w, generator=generator)
    raise ValueError(kind)

def generate(kind, steps, labels, w, seed):
    # batched generation of a whole label set; returns images on CPU and seconds taken
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.time(); out = []
    for i in range(0, len(labels), CFG['sample_batch']):
        out.append(sample(kind, ema, sched, labels[i:i + CFG['sample_batch']], steps, w, g).cpu())
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return torch.cat(out), time.time() - t0

from PIL import Image, ImageDraw, ImageFont

def _font(size, bold=False):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 'DejaVuSans.ttf'] + (['arialbd.ttf'] if bold else []) + ['arial.ttf']:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()

ONES = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve',
        'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
TENS = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
def _two(n): return ONES[n] if n < 20 else (TENS[n // 10] + (' ' + ONES[n % 10] if n % 10 else ''))
def _three(n): return (ONES[n // 100] + ' Hundred' + (' ' if n % 100 else '') if n >= 100 else '') + (_two(n % 100) if n % 100 else '')
def amount_in_words(n):
    # Indian numbering system: crore, lakh, thousand
    if n == 0: return 'Zero'
    parts = []
    for div, name in [(10**7, 'Crore'), (10**5, 'Lakh'), (10**3, 'Thousand')]:
        if n >= div: parts.append(f'{_three(n // div) if div == 10**7 else _two(n // div)} {name}'); n %= div
    if n: parts.append(_three(n))
    return ' '.join(parts)

def render_cheque(digit_imgs, amount, rng):
    # digit_imgs: tensor [n,1,28,28] in [-1,1] -> PIL cheque image
    W, H = 1100, 480
    img = Image.new('RGB', (W, H), (246, 250, 244)); d = ImageDraw.Draw(img)
    for y in range(0, H, 6): d.line([(0, y), (W, y)], fill=(236, 243, 233))            # security-print texture
    d.rectangle([8, 8, W - 9, H - 9], outline=(60, 110, 80), width=3)
    d.text((30, 24), 'SAMPLE BANK OF EXAMPLES (FICTIONAL)', font=_font(26, True), fill=(40, 80, 60))
    d.text((30, 58), 'Demo branch · IFSC: XXXX0000000 · for synthetic-data generation only', font=_font(15), fill=(90, 110, 100))
    d.text((W - 260, 30), 'Date  DD/MM/YYYY', font=_font(18), fill=(60, 60, 60))
    d.text((30, 130), 'Pay', font=_font(20), fill=(60, 60, 60)); d.line([(80, 152), (W - 60, 152)], fill=(120, 120, 120))
    d.text((90, 124), 'ChequeGen Synthetic Payee', font=_font(20), fill=(30, 30, 90))
    d.text((30, 190), 'Rupees', font=_font(20), fill=(60, 60, 60)); d.line([(110, 212), (700, 212)], fill=(120, 120, 120))
    d.text((118, 186), amount_in_words(amount) + ' Only', font=_font(18), fill=(30, 30, 90))
    bx0, by0, bx1, by1 = 730, 240, W - 40, 330                                            # amount box
    d.rectangle([bx0, by0, bx1, by1], outline=(60, 60, 60), width=2, fill=(255, 255, 255))
    d.text((bx0 + 10, by0 + 22), '₹', font=_font(40, True), fill=(60, 60, 60))
    x = bx0 + 60
    for i, dimg in enumerate(digit_imgs):
        a = ((1 - (dimg[0].clamp(-1, 1) + 1) / 2).numpy() * 255).astype(np.uint8)      # white paper, dark ink
        size = int(rng.integers(58, 70))
        tile = Image.fromarray(a).resize((size, size), Image.BICUBIC)
        ink = Image.new('RGB', tile.size, (20, 30, 110))                                  # blue-black pen
        mask = Image.fromarray(255 - np.array(tile)).point(lambda v: 0 if v < 40 else v)
        img.paste(ink, (x, by0 + 12 + int(rng.integers(-6, 7))), mask)
        x += int(size * 0.62)
        if x > bx1 - 40: break
    d.text((30, 390), 'A/c No. 000000000000 (dummy)', font=_font(16), fill=(90, 90, 90))
    d.text((W - 300, 400), 'Authorised Signatory', font=_font(16), fill=(90, 90, 90))
    # SPECIMEN watermark: makes clear this is not a usable cheque
    wm = Image.new('RGBA', (W, H), (0, 0, 0, 0)); wd = ImageDraw.Draw(wm)
    wd.text((170, 150), 'SPECIMEN', font=_font(170, True), fill=(200, 40, 40, 70))
    img = Image.alpha_composite(img.convert('RGBA'), wm.rotate(12, center=(W / 2, H / 2))).convert('RGB')
    ImageDraw.Draw(img).text((W - 420, H - 34), 'SPECIMEN · synthetic data · not a valid instrument', font=_font(14), fill=(180, 40, 40))
    return img

def generate_cheque(net, sched, amount_text, steps=20, w=1.5, seed=0, device='cpu', sampler='dpm'):
    digits = [int(c) for c in str(amount_text) if c.isdigit()][:9]
    if not digits: raise ValueError('Enter a numeric amount, e.g. 45750')
    y = torch.tensor(digits, device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    t0 = time.time()
    imgs = (dpm_solver_pp_sample(net, sched, y, steps, w, generator=g) if sampler == 'dpm' else ddim_sample(net, sched, y, steps, w, generator=g)).cpu()
    secs = time.time() - t0
    amount = int(''.join(map(str, digits)))
    return render_cheque(imgs, amount, np.random.default_rng(int(seed))), secs
