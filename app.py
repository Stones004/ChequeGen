"""ChequeGen — Gradio demo.

Type an amount; a class-conditional diffusion model writes it in handwriting on a SPECIMEN cheque
of a fictional bank, using fast DPM-Solver++ or DDIM sampling (or the slow DDPM baseline, for comparison).

Run:  pip install -r requirements.txt && python app.py
Needs checkpoints/chequegen_ema.pt (created by ChequeGen.ipynb).
"""
import os, sys, time
import numpy as np
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import chequegen as cg  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = os.path.join(os.path.dirname(__file__), 'checkpoints', 'chequegen_ema.pt')

ckpt = torch.load(CKPT, map_location=DEVICE)
cfg = ckpt['cfg']
BEST_W = float(ckpt.get('best_w', cfg.get('guidance', 1.5)))
model = cg.CondUNet(ch=tuple(cfg['channels'])).to(DEVICE)
model.load_state_dict(ckpt['ema'])
model.eval()
sched = cg.Schedule(cfg['T'], cfg['beta_start'], cfg['beta_end'], DEVICE)

SAMPLERS = {'DPM-Solver++ (fastest)': 'dpm', 'DDIM': 'ddim', 'DDPM baseline (1000 steps, slow)': 'ddpm'}


def run(amount, sampler_name, steps, guidance, seed):
    kind = SAMPLERS[sampler_name]
    if kind == 'ddpm':
        digits = [int(c) for c in str(amount) if c.isdigit()][:9]
        if not digits:
            raise gr.Error('Enter a numeric amount, e.g. 45750')
        y = torch.tensor(digits, device=DEVICE)
        g = torch.Generator(device=DEVICE).manual_seed(int(seed))
        t0 = time.time()
        imgs = cg.ddpm_sample(model, sched, y, float(guidance), generator=g).cpu()
        secs = time.time() - t0
        img = cg.render_cheque(imgs, int(''.join(map(str, digits))), np.random.default_rng(int(seed)))
        return img, f'DDPM baseline: {secs:.2f} s for 1000 steps.'
    img, secs = cg.generate_cheque(model, sched, amount, int(steps), float(guidance), int(seed), DEVICE, kind)
    return img, f'{sampler_name.split(" (")[0]}: {secs:.2f} s for {int(steps)} steps.'


with gr.Blocks(title='ChequeGen') as demo:
    gr.Markdown(
        '# ChequeGen\n'
        'Fast diffusion handwriting for privacy-safe cheque-OCR training data. '
        'Compare the slow **DDPM baseline** with the fast **DDIM** and **DPM-Solver++** samplers. '
        'Output is watermarked **SPECIMEN** and uses a fictional bank.'
    )
    with gr.Row():
        with gr.Column(scale=1):
            amount = gr.Textbox(value='45750', label='Amount (₹, digits only)')
            sampler = gr.Radio(list(SAMPLERS), value='DPM-Solver++ (fastest)', label='Sampler')
            steps = gr.Slider(5, 100, value=20, step=5, label='Sampling steps (ignored for DDPM)')
            guidance = gr.Slider(0, 4, value=BEST_W, step=0.5, label=f'Guidance weight w (tuned: {BEST_W})')
            seed = gr.Number(value=1, precision=0, label='Writer seed (changes the handwriting style)')
            btn = gr.Button('Write cheque', variant='primary')
        with gr.Column(scale=2):
            out_img = gr.Image(label='Synthetic SPECIMEN cheque', type='pil')
            out_txt = gr.Textbox(label='Timing')
    btn.click(run, [amount, sampler, steps, guidance, seed], [out_img, out_txt])
    gr.Examples([['45750', 'DPM-Solver++ (fastest)', 20, BEST_W, 1], ['120000', 'DDIM', 50, BEST_W, 7],
                 ['9876', 'DPM-Solver++ (fastest)', 5, BEST_W, 3]], [amount, sampler, steps, guidance, seed])

if __name__ == '__main__':
    demo.launch()
