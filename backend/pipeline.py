# backend/pipeline.py
from pathlib import Path
import tempfile, zipfile, csv, json, textwrap
from datetime import datetime

# ---- OpenAI helper (reads key from st.secrets) ---------------------------
from backend.openai_utils import chat, MAX_PATIENTS

# -------------------------------------------------------------------------
def generate_cohort(
    icd_code: str,
    icd_label: str,
    files,                    # list of UploadedFile objects from Streamlit
    comments: str,            # "\n"-separated scope notes
    n: int,
    demo_filters: dict,
    seed: str | None = None,
    benchmark=None,
):
    """
    Return a ZIP path + run-id for a synthetic-patient cohort.

    • Uses OpenAI to create the patient rows.
    • Keeps everything in a temp dir so Streamlit can stream the ZIP out.
    """

    # ---------------------------- guard-rails -----------------------------
    if n > MAX_PATIENTS:               # defined in .streamlit/secrets.toml
        raise ValueError(f"n ({n}) exceeds MAX_PATIENTS={MAX_PATIENTS}")
    if not icd_code or not icd_label:
        raise ValueError("ICD-10 code/label missing")

    # ----------------------  build the LLM prompt  -----------------------
    prompt_parts = [
        f"You are a clinical data engine that fabricates HIGH-QUALITY "
        f"synthetic patients strictly for software testing and demo.",
        "",
        f"Diagnosis to model: **{icd_label}** (ICD-10 {icd_code}).",
        f"Number of patients requested: **{n}**.",
    ]

    # demographic filters --------------------------------------------------
    if demo_filters:
        readable = ", ".join(
            f"{k.replace('_',' ')}={v}" for k, v in demo_filters.items() if v
        )
        if readable:
            prompt_parts.append(f"Apply these demographic constraints: {readable}")

    # scope notes + VERY short snippets of uploaded docs -------------------
    if files:
        notes = [ln.strip() for ln in comments.splitlines()]
        doc_blurbs = []
        for idx, file in enumerate(files):
            label = notes[idx] if idx < len(notes) else file.name
            # read at most first ~2k chars – just enough for context
            snippet = file.read(2048).decode(errors="ignore")
            file.seek(0)
            doc_blurbs.append(
                f"\n### {label}\n{snippet}\n"
            )
        prompt_parts.append(
            "The following reference material MAY inform comorbidities, meds, "
            "or lab patterns – do not copy text verbatim:\n"
            + "\n".join(doc_blurbs)
        )

    # instruct JSON return format -----------------------------------------
    prompt_parts.append(
        textwrap.dedent(
            """
            Respond ONLY with valid JSON – an array of objects.  
            Each object must contain:

            - patient_id          (string, unique)  
            - age                 (integer)  
            - sex                 ("F" or "M")  
            - race                (US CDC wide-band)  
            - ethnicity           ("Hispanic or Latino" / "Not Hispanic or Latino")  
            - icd10_code          (string)  
            - diagnosis           (string)

            Return NO markdown fences, NO commentary – pure JSON.
            """
        )
    )

    system_msg = {"role": "system", "content": "You are a careful medical data generator."}
    user_msg   = {"role": "user",   "content": "\n".join(prompt_parts)}

    # --------------------------- OpenAI call ------------------------------
    response = chat([system_msg, user_msg])
    raw_json = response.choices[0].message.content.strip()

    try:
        patients = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}") from None

    # --------------------------- CSV output -------------------------------
    tmp_dir   = Path(tempfile.mkdtemp())
    csv_path  = tmp_dir / "patients.csv"
    headers   = ["patient_id", "age", "sex", "race", "ethnicity",
                 "icd10_code", "diagnosis"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in patients:
            writer.writerow(row)

    # ----------------------------- ZIP it ---------------------------------
    zip_path = tmp_dir / "cohort.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, csv_path.name)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return zip_path, run_id
