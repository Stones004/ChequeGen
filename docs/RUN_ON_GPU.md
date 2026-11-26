# Running ChequeGen v2 on your local NVIDIA GPU (Windows)

The notebook detects the GPU and switches to the large configuration on its own: a 3.7 M-parameter U-Net, 20,000 training steps with mixed precision, and 2,000 images × 3 seeds per sampler. Expect about **30–60 minutes** on a laptop RTX GPU.

## 1. Check the GPU and driver
```powershell
nvidia-smi
```
Note the **CUDA Version** in the top-right corner. It is the highest CUDA version your driver supports.

## 2. Create an environment and install PyTorch with CUDA
Run these in the unzipped `ChequeGen` folder:
```powershell
python -m venv .venv
.venv\Scripts\activate
# driver shows CUDA 12.4 or newer:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# driver shows CUDA 12.1-12.3: use .../whl/cu121 instead
pip install -r requirements.txt
```
Check that PyTorch can see the GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It must print `True` and your GPU's name.

## 3. Run the whole notebook
Run it either way:
* **VS Code / Jupyter:** open `ChequeGen.ipynb`, select the `.venv` kernel, then **Run All**.
* **Command line:** this saves all outputs into the notebook:
```powershell
jupyter nbconvert --to notebook --execute ChequeGen.ipynb --output ChequeGen.ipynb --ExecutePreprocessor.timeout=-1
```
The CPU checkpoint in `checkpoints/` has a different model size. The notebook notices this and retrains on the GPU by itself, so you don't need to delete anything.

**If you see "CUDA out of memory":** in the setup cell, change `sample_batch=500 if GPU` to `250` (or `batch_size=256` to `128`), then run again.

## 4. Send the results back
Send these files:
* `ChequeGen.ipynb` (executed)
* `results.json`
* the `figures/` folder

All interpretation text in the notebook is generated from the run's own numbers, so it stays correct for the GPU results. With the files above, the project document and the PDFs can be updated to match.

## 5. Try the app
```powershell
python app.py
```
Then open http://127.0.0.1:7860.
