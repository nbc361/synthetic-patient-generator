# app.py  ──────────────────────────────────────────────────────────────────
# Streamlit front-end for the Synthetic Patient Cohort Generator
# -------------------------------------------------------------------------

import streamlit as st
import requests

# back-end helpers
from backend.pipeline import generate_cohort, _parse_extra_schema

st.set_page_config(page_title="Patient Cohort Generator", layout="centered")
st.title("Patient Cohort Generator")

# ─────────────────────────── ICD-10 lookup ───────────────────────────────
code      = st.text_input("ICD-10 code (e.g. J47, L73.2, K50)")
diagnosis = ""

if code:
    raw = code.strip().upper()
    url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
    try:
        data = requests.get(
            url,
            params={"sf":"code,name","df":"name","terms":raw,"maxList":50},
            timeout=5
        ).json()                              # [count,[codes],null,[[name]…],…]
        count, codes, names = data[0], data[1], data[3]

        idx = next((i for i,c in enumerate(codes) if c.upper().startswith(raw)), None)
        if count == 0 or idx is None:
            st.error("ICD-10 code not found."); st.stop()

        full_code  = codes[idx]
        diagnosis  = names[idx][0]
        st.success(f"**{full_code} — {diagnosis}**")
    except Exception as e:
        st.error(f"Lookup error: {e}")

# ───────────────────── documents + scope notes ───────────────────────────
files = st.file_uploader(
    "Upload PDFs / DOCX / TXT",
    accept_multiple_files=True,
    type=["pdf","docx","txt"]
)
comments = st.text_area(
    "Scope note for each file",
    help="Enter one line per file (same order as upload)."
)

# ───────────────────────── cohort size slider ────────────────────────────
N = st.slider("Patient population size", 1, 100, 25)

# ─────────────────────────── demographics box ────────────────────────────
with st.expander("Optional demographic filters"):
    race   = st.multiselect("Race", [
        "White","Black or African American","Asian",
        "American Indian / Alaska Native","Native Hawaiian / Pacific Islander","Other"])
    eth    = st.multiselect("Ethnicity",
        ["Hispanic or Latino","Not Hispanic or Latino"])
    gender = st.multiselect("Gender", ["Female","Male","Non-binary / Other"])
    age_min, age_max = st.slider("Age range", 0, 100, (0, 100))

# ───────────────── extra dynamic column definitions ──────────────────────
with st.expander("Optional extra patient attributes"):
    st.markdown(
        """
**Define extra columns**

* One line per column → `field_name : type`

**Allowed types**

| Type | What the model will generate | Example values |
|------|------------------------------|----------------|
| `int`   | whole numbers              | `0`, `3`, `42` |
| `float` | decimal numbers            | `0.0`, `98.6`, `1.25` |
| `str`   | free-text strings          | `"Yes"`, `"Ex-smoker"` |

*(≈ 10 columns max is a good rule of thumb.)*

**Example**
fev1_pct : float
sputum_volume_ml : int
smoking_status : str
"""
    )

    extra_schema_txt = st.text_area(
        "Schema lines",
        height=160,
        placeholder="fev1_pct : float",
    )

    if extra_schema_txt.strip():
        try:
            cols = _parse_extra_schema(extra_schema_txt)
            st.success("✓ Detected columns " + ", ".join(nm for nm, _ in cols))
        except Exception as e:
            st.error(f"⚠️ {e}")

# ───────────────────────── advanced options ──────────────────────────────
with st.expander("Advanced"):
    seed_txt = st.text_input("Run seed (blank = random)", "")
    bench    = st.file_uploader("Optional benchmark CSV", type="csv", key="bench")

# ───────────────────────────── generate! ─────────────────────────────────
if st.button("Generate cohort", type="primary"):

    if not code or not diagnosis:
        st.warning("Please enter a valid ICD-10 code."); st.stop()

    if files and len([ln for ln in comments.splitlines() if ln.strip()]) != len(files):
        st.warning("Comment lines don’t match number of files."); st.stop()

    with st.spinner("Running AI pipeline – this may take a minute…"):
        out_zip, run_id = generate_cohort(
            icd_code      = code.upper(),
            icd_label     = diagnosis,
            files         = files,
            comments      = comments,
            n             = N,
            demo_filters  = dict(
                race=race, ethnicity=eth, gender=gender, age=(age_min, age_max)
            ),
            seed          = seed_txt,
            benchmark     = bench,
            extra_schema  = extra_schema_txt      # ← pass extra-column schema
        )

    st.success(f"Cohort ready!  **Run ID: {run_id}**")
    st.download_button(
        "Download ZIP (CSV + meta)",
        data   = open(out_zip, "rb"),
        file_name = f"cohort_{run_id}.zip",
        mime  = "application/zip",
    )
    
