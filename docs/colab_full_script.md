# Colab — full script (copy-paste)

Open [notebooks/cmapss_colab_train_all.ipynb](../notebooks/cmapss_colab_train_all.ipynb) in Colab (**File → Upload notebook**), or paste the cells below.

**Set your repo URL in the first code cell**, then run all cells.

---

## Cell 1 — Config

```python
REPO_URL = "https://github.com/YOUR_USERNAME/predictive-maintenance.git"
REPO_DIR = "/content/predictive-maintenance"
BRANCH = "main"
TRAIN_MODE = "fast"  # "fast" or "full"
LSTM_EPOCHS = 10
```

---

## Cell 2 — Clone

```python
import shutil
from pathlib import Path

if "YOUR_USERNAME" in REPO_URL:
    raise ValueError("Set REPO_URL to your GitHub HTTPS URL.")

if Path(REPO_DIR).exists():
    shutil.rmtree(REPO_DIR)

!git clone --branch {BRANCH} --depth 1 {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
print("CWD:", Path.cwd())
```

---

## Cell 3 — Install

```python
%pip install -q torch --index-url https://download.pytorch.org/whl/cu124
%pip install -q -r requirements.txt
```

```python
import torch
from pathlib import Path
assert Path("scripts/cmapss_colab_train.py").exists()
print("GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## Cell 4 — Download CMAPSS

```python
import urllib.request
from pathlib import Path

RAW = Path("data/raw/cmapss")
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://raw.githubusercontent.com/kpzhang93/DTAFM/master/CMAPSSData"

for fd in ["FD001", "FD002", "FD003", "FD004"]:
    for prefix in ("train", "test", "RUL"):
        name = f"{prefix}_{fd}.txt"
        dest = RAW / name
        if not dest.exists() or dest.stat().st_size < 100:
            urllib.request.urlretrieve(f"{BASE}/{name}", dest)
        print("ok", name)
```

---

## Cell 5 — Train (Phase 2 + Phase 3 + MLflow)

```python
import os
os.environ["MLFLOW_TRACKING_URI"] = "./mlruns"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "predictive_maintenance"

if TRAIN_MODE == "fast":
    !python scripts/cmapss_colab_train.py --fast
else:
    !python scripts/cmapss_colab_train.py --lstm-epochs {LSTM_EPOCHS}
```

---

## Cell 6 — Verify

```python
!python scripts/report_cmapss_mlflow.py
```

```python
import json
from pathlib import Path
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file://" + str(Path.cwd() / "mlruns"))
exp = MlflowClient().get_experiment_by_name("predictive_maintenance")
runs = MlflowClient().search_runs([exp.experiment_id], max_results=50)
rows = [{
    "dataset": r.data.params.get("dataset_id"),
    "winner": r.data.params.get("winner"),
    "test_rmse": r.data.metrics.get("test_rmse"),
    "test_nasa": r.data.metrics.get("test_rul_score"),
} for r in runs if (r.info.run_name or "").endswith("_phase3_summary")]
display(pd.DataFrame(rows).sort_values("dataset"))
```

---

## Cell 7 — Download zip

```python
from pathlib import Path
from google.colab import files

!zip -qr cmapss_colab_outputs.zip mlruns models artifacts data/processed
files.download("cmapss_colab_outputs.zip")
```

---

## After download (local PC)

Unzip into your project root, then:

```bash
python scripts/report_cmapss_mlflow.py
mlflow ui --backend-store-uri ./mlruns
streamlit run dashboard/app.py
```
