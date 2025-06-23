import streamlit as st
from backend.bias_eval import run_bias

st.header("Bias & Fairness Report")

gen_csv   = st.file_uploader("Generated cohort CSV", type="csv")
bench_csv = st.file_uploader("Benchmark CSV",         type="csv")

if st.button("Run report") and gen_csv and bench_csv:
    with st.spinner("Crunching numbers…"):
        result = run_bias(gen_csv, bench_csv)
    st.dataframe(result)
    st.success("Done – scroll table for parity ratios ≤ 0.8 or ≥ 1.25.")
