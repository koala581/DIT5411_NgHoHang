# Complete Windows + WSL2 Ubuntu setup (English)

This guide assumes a **new Windows computer** and a supported NVIDIA GPU. All terminal instructions are intended to be copied into the correct shell, as indicated below. The final project is in English.

## 1. Install/update NVIDIA graphics drivers (Windows)

Install an appropriate Windows NVIDIA GPU driver from the official NVIDIA website if necessary. Open **Windows PowerShell** and run:

```powershell
nvidia-smi
```

Confirm that an NVIDIA GPU is displayed. Do not install the Linux NVIDIA display driver inside WSL.

## 2. Install WSL2 Ubuntu (Windows PowerShell as Administrator)

```powershell
wsl --install -d Ubuntu
```

Restart Windows **only if prompted**. To check the installed distribution and WSL version:

```powershell
wsl --list --verbose
```

If Ubuntu has `VERSION 1`, run `wsl --set-version Ubuntu 2`. Launch Ubuntu with:

```powershell
wsl -d Ubuntu
```

During the first launch, create a Linux username and password. The password does not display characters while being typed. **Important:** the installed distro is called `Ubuntu`; use `wsl -d Ubuntu`, not `wsl -d Ubuntu-24.04`, unless that is the name actually shown by `wsl --list --verbose`.

## 3. Verify GPU visibility in Ubuntu

In **Ubuntu Terminal**:

```bash
nvidia-smi
```

This checks WSL's NVIDIA driver access, but it is **not** yet the TensorFlow GPU check.

## 4. Install tools, Python 3.12 and a virtual environment (Ubuntu)

```bash
sudo apt update
sudo apt install -y curl ca-certificates git unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
uv python install 3.12
mkdir -p ~/DIT5411
cd ~/DIT5411
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
```

Using `uv` avoids Ubuntu's potentially incompatible system Python (for example, Python 3.14). `uv`-created virtual environments may not have the traditional `pip` module; use `uv pip`, as shown below.

## 5. Install TensorFlow GPU and JupyterLab

```bash
uv pip install 'tensorflow[and-cuda]==2.20.0' numpy pandas matplotlib jupyterlab ipykernel
```

TensorFlow **2.20.0** is pinned to avoid a reported 2.21.0 GPU library-path regression. The hardware/software combination still needs to be verified on this PC.

```bash
python -c 'import tensorflow as tf; print("TF version:", tf.__version__); print("GPUs:", tf.config.list_physical_devices("GPU"))'
```

The GPU list must contain a `PhysicalDevice(...GPU...)`, not `[]`. If it is empty, troubleshoot the NVIDIA driver, WSL version, and CUDA dependencies **before** training. Installing NVIDIA drivers inside WSL is usually unnecessary and can break WSL GPU setup.

Register the Jupyter kernel:

```bash
python -m ipykernel install --user --name dit5411 --display-name 'Python (DIT5411)'
```

## 6. Download the complete 13,065-character dataset (Ubuntu)

Work inside the Linux filesystem rather than `/mnt/c` for this image-heavy project.

```bash
cd ~/DIT5411
df -h .
git clone --depth 1 https://github.com/chenkenanalytic/handwritting_data_all.git dataset_source
ls -lh dataset_source/all_data.zip.*
cat dataset_source/all_data.zip.0?? > all_data.zip
unzip -t all_data.zip | tail -n 2
mkdir -p dataset
unzip -O big5 -q all_data.zip -d dataset
find dataset/cleaned_data -mindepth 1 -maxdepth 1 -type d | wc -l
```

You should see **13,065** class folders. The notebook checks **684,677** images. The archive is large: leave sufficient disk space for the source repo, merged ZIP, extracted images and model checkpoints. Once extraction is verified, `all_data.zip` may be removed to reclaim disk space; keep `dataset/cleaned_data`.

## 7. Copy and open the notebook

From Ubuntu (inside `~/DIT5411`), run `explorer.exe .` to open this Linux folder in Windows File Explorer. Copy `DIT5411_Three_Models_Step_by_Step.ipynb` and the English README files to it.

Then, in Ubuntu:

```bash
cd ~/DIT5411
source .venv/bin/activate
jupyter lab --no-browser
```

Open the `http://localhost:8888/...` link shown in the terminal using your Windows browser. Select the kernel **Python (DIT5411)**, and run the notebook **top to bottom**.

## 8. Run the full three-model experiment

The notebook has 18 numbered code steps, grouped into preparation, augmentation, model definitions, three separate training cells, evaluation and reporting:

1. Steps 1–3: verify GPU, audit all 13,065 classes, and create `train.csv` and `test.csv`.
2. Steps 4–8: create 200 training samples/class, preprocess images, and build TensorFlow pipelines.
3. Steps 9–11: define the MLP, Basic CNN and Deep CNN.
4. Steps 12–15: set up checkpoints, then **train each model in its own cell with `verbose=1` batch/step progress**.
5. Steps 16–18: evaluate the three models, plot curves, and export the English README results table.

`EPOCHS=5`, `BATCH_SIZE=32`, and `IMG_SIZE=64` are the defaults. Reduce the batch size to 16 if the GPU runs out of memory; increase it only if memory permits. The workload is large: **2,613,000 training examples per model per epoch**, and training all three models can take substantial time.

The notebook writes `results/models/*_best.keras`, `results/*_history.csv`, `results/model_comparison.csv`, `results/model_comparison.png`, `results/training_samples_preview.png`, and `results/RESULTS_FOR_README.md`. A completed model is skipped if its `.done` marker and best checkpoint exist. After an interruption, model training can resume from the last saved *completed epoch* using `BackupAndRestore` when the previous backup files remain. A partially completed epoch may need to run again.

## 9. Submit the English GitHub repository

1. Create a **public GitHub repository** using the exact assignment naming convention `DIT5411-your name` (replace `your name` with your actual required name).
2. Upload the `.ipynb`, `README.md`, `SETUP_EN.md`, `requirements.txt`, `.gitignore`, and **actual** CSV/PNG results.
3. Open `results/RESULTS_FOR_README.md` and paste its measured accuracy table and chart link into the README.
4. Do **not** upload the hundreds of thousands of original images, the source ZIPs, `.venv`, model checkpoints or backup files. The `.gitignore` excludes them.
5. **Clear notebook cell outputs before the final commit** to keep the `.ipynb` small. This does not affect the on-screen step progress while training, or the saved CSV/charts. In JupyterLab use **Edit → Clear All Outputs** or the equivalent menu option.
6. Submit the GitHub **repository URL**, not only the `.ipynb` file.

## 10. Common troubleshooting

- `wsl -d Ubuntu-24.04` shows `WSL_E_DISTRO_NOT_FOUND`: check the registered name using `wsl --list --verbose`; it may be `Ubuntu`.
- `ensurepip is not available` / `No module named pip`: use `uv python install 3.12`, `uv venv --python 3.12 .venv`, then install with `uv pip`.
- `Cannot dlopen some GPU libraries` / `GPU: []`: check you installed packages **inside the activated WSL virtual environment**, that the GPU is visible with `nvidia-smi`, and that the TensorFlow version is pinned correctly.
- `ResourceExhaustedError` / CUDA out of memory: restart Jupyter kernel, lower `BATCH_SIZE` to 16, and rerun cells; don't compare runs with different configurations without documenting the change.
- The notebook is large: Jupyter saves generated progress output. After results have been written to disk, **clear cell outputs and save the notebook** before GitHub submission.
