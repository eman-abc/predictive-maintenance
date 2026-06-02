# Colab — full script (upload CMAPSS to SSD)

Open [notebooks/cmapss_colab_train_all.ipynb](../notebooks/cmapss_colab_train_all.ipynb) in Colab.

Put the 12 NASA files on Colab disk at **`/content/cmapss_upload/`**, then run the notebook.

---

## Cell 1 — Config

```python
REPO_URL = "https://github.com/eman-abc/predictive-maintenance.git"
REPO_DIR = "/content/predictive-maintenance"
BRANCH = "main"
TRAIN_MODE = "fast"
LSTM_EPOCHS = 10
CMAPSS_UPLOAD_DIR = "/content/cmapss_upload"
```

---

## Cell 2 — Clone

```python
import shutil
from pathlib import Path

if Path(REPO_DIR).exists():
    shutil.rmtree(REPO_DIR)
!git clone --branch {BRANCH} --depth 1 {REPO_URL} {REPO_DIR}
%cd {REPO_DIR}
```

---

## Cell 3 — Install

```python
%pip install -q torch --index-url https://download.pytorch.org/whl/cu124
%pip install -q -r requirements.txt
```

---

## Cell 4 — Prepare upload folder on Colab SSD

```python
from pathlib import Path
UPLOAD_DIR = Path(CMAPSS_UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
print(UPLOAD_DIR, list(UPLOAD_DIR.glob("*.txt")))
```

---

## Cell 5 — Upload from your PC (file picker)

```python
from google.colab import files
uploaded = files.upload()  # select all 12 .txt files
for name, data in uploaded.items():
    (UPLOAD_DIR / name).write_bytes(data)
    print(name, len(data))
```

**Or** use Colab left sidebar **Files** → upload into `/content/cmapss_upload/` and skip this cell.

---

## Cell 6 — (Optional) Unzip NASA archive

```python
import zipfile
for z in Path("/content").glob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(UPLOAD_DIR)
```

---

## Cell 7 — Import into project

```python
!python scripts/import_cmapss_upload.py --source /content/cmapss_upload
```

---

## Cell 8 — Train (MLflow)

```python
import os
os.environ["CMAPSS_UPLOAD_DIR"] = CMAPSS_UPLOAD_DIR
!python scripts/cmapss_colab_train.py --fast --upload-dir /content/cmapss_upload
```

---

## Cell 9 — Verify + download zip

```python
!python scripts/report_cmapss_mlflow.py
```

```python
from google.colab import files
!zip -qr cmapss_colab_outputs.zip mlruns models artifacts data/processed
files.download("cmapss_colab_outputs.zip")
```
