# LatentPre

This repository provides the implementation for ***Fair Data Pre-Processing with Imperfect Attribute Space***, accepted at SIGMOD 2026.

## 1. Setup

* Create a new environment in conda: 
```bash
conda create -n repair python=3.8
```

* Activate environment: 
```bash
conda activate repair
```

* Install dependencies:
```bash
conda install numpy>=1.22 pandas scipy matplotlib scikit-learn numba psutil seaborn statsmodels
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install conda-forge tabulate pot metric-learn fairlearn memory_profiler pexpect pickleshare backcall decorator wcwidth
pip install python-sat
```

## 2. Run
```bash
python main.py
```

