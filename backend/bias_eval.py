"""
Compute Aequitas fairness metrics for a generated cohort vs. benchmark.
"""

from aequitas.group import Group
from aequitas.bias import Bias
import pandas as pd

KEY_COL    = "patient_id"   # used just for row counts
TARGET_COL = "icd10_code"   # we only have positives (all rows)

def run_bias(gen_csv: str, bench_csv: str, attributes=("race", "ethnicity", "sex")):
    gen_df   = pd.read_csv(gen_csv)
    bench_df = pd.read_csv(bench_csv)

    gen_df["label_value"]   = 1                # all generated rows are “positive”
    bench_df["label_value"] = 1

    g = Group()(gen_df, attributes)
    b = Bias().get_bias(g, perf_metric="pprev")   # positive-prevalence parity
    return b
