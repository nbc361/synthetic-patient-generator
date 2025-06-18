# backend/pipeline.py
from pathlib import Path
import tempfile
import zipfile
import csv
import random
from datetime import datetime

def generate_cohort(
        icd_code: str,
        icd_label: str,
        files,
        comments: str,
        n: int,
        demo_filters: dict,
        seed: str|None = None,
        benchmark=None):
    """
    Dummy generator that just spits out a CSV with n fake rows.
    Replace this with your real logic later.
    """
    if seed:
        random.seed(seed)

    # create a temp CSV
    tmp_dir = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / "patients.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "icd10_code", "diagnosis"])
        for i in range(1, n + 1):
            writer.writerow([f"P{i:04d}", icd_code, icd_label])

    # zip it up
    zip_path = tmp_dir / "cohort.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, csv_path.name)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return zip_path, run_id
