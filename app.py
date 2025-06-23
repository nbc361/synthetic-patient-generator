# app.py  ────────────── Streamlit front-end (UI only)

import streamlit as st
import requests
from backend.pipeline import generate_cohort      # <- we’ll create this soon

st.set_page_config(page_title="Patient Cohort Generator", layout="centered")
st.title("Patient Cohort Generator")

# ── ICD-10 lookup ──────────────────────────────────────────────────────────
code = st.text_input("ICD-10 code (e.g. J47, L73.2, K50)")
diagnosis = ""

if code:
    raw = code.strip().upper()
    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    params = {"sf": "code,name",       # search both columns
              "df": "name",            # display only the name
              "terms": raw,
              "maxList": 50}           # (optional) widen result pool

    try:
        resp  = requests.get(url, params=params, timeout=5)
        data  = resp.json()            # [count, [codes], null, [[name] …], …]
        count = data[0]
        codes = data[1]
        names = data[3]                # <-- descriptions live here

        if count == 0:
            st.error("ICD-10 code not found.")
            st.stop()

        # first result whose code starts with the user’s text
        match_idx = next((i for i, c in enumerate(codes)
                          if c.upper().startswith(raw)), None)

        if match_idx is None:
            st.error("ICD-10 code not found.")
            st.stop()

        full_code = codes[match_idx]
        diagnosis = names[match_idx][0]     # names[i] is a 1-element list
        st.success(f"**{full_code} — {diagnosis}**")

    except Exception as e:
        st.error(f"Lookup error: {e}")
        
# ── Upload docs + comments ────────────────────────────────────────────────
files = st.file_uploader("Upload PDFs / DOCX / TXT", accept_multiple_files=True,
                         type=["pdf", "docx", "txt"])
comment_help = ("Enter *one line per file* describing its focus, "
                "in the same order you uploaded the files.")
comments = st.text_area("Scope note for each file", help=comment_help)

# ── Population size ───────────────────────────────────────────────────────
N = st.slider("Patient population size", 1, 100, 25)

# ── Demographic filters (collapsible) ─────────────────────────────────────
with st.expander("Optional demographic filters"):
    race    = st.multiselect("Race", ["White","Black or African American","Asian",
                                      "American Indian / Alaska Native",
                                      "Native Hawaiian / Pacific Islander","Other"])
    eth     = st.multiselect("Ethnicity", ["Hispanic or Latino","Not Hispanic or Latino"])
    gender  = st.multiselect("Gender", ["Female","Male","Non-binary / Other"])
    age_min, age_max = st.slider("Age range", 0, 100, (0, 100))

# ── Advanced box (seed + benchmark) ───────────────────────────────────────
with st.expander("Advanced"):
    seed_txt = st.text_input("Run seed (blank = random)", "")
    bench    = st.file_uploader("Optional benchmark CSV", type="csv", key="bench")

# ── Generate button ───────────────────────────────────────────────────────
if st.button("Generate cohort"):

    # quick validations
    if not code or not diagnosis:
        st.warning("Please enter a valid ICD-10 code.")
        st.stop()
    if files and (len([l for l in comments.splitlines() if l.strip()]) != len(files)):
        st.warning("Comment lines don’t match number of files.")
        st.stop()

    # call the back-end pipeline
    with st.spinner("Running AI pipeline – this may take a minute…"):
        out_zip, run_id = generate_cohort(
            icd_code=code.upper(),
            icd_label=diagnosis,
            files=files,
            comments=comments,
            n=N,
            demo_filters=dict(race=race, ethnicity=eth, gender=gender,
                              age=(age_min, age_max)),
            seed=seed_txt,
            benchmark=bench)

    st.success(f"Cohort ready!  **Run ID: {run_id}**")
    st.download_button("Download ZIP (CSV + meta)",
                       data=open(out_zip, "rb"),
                       file_name=f"cohort_{run_id}.zip",
                       mime="application/zip")
