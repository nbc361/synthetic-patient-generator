# backend/pipeline.py
"""End-to-end cohort generator."""

from pathlib import Path
from datetime import datetime
import tempfile, zipfile, csv, json, textwrap, random

# ── local helpers ---------------------------------------------------------
from backend.openai_utils  import chat, MAX_PATIENTS          # OpenAI wrapper
from backend.data_ingest   import ingest                      # your new ingester

# ───────────────────────────────────────────────────────────────────────────
def generate_cohort(
    icd_code:      str,
    icd_label:     str,
    files,                       # list[UploadedFile] from Streamlit
    comments:      str,          # “\n”-separated scope notes
    n:             int,
    demo_filters:  dict,
    seed:          str | None = None,
    benchmark                   = None,
):
    """
    Build a synthetic-patient CSV + ZIP and return (zip_path, run_id).

    • Indexes any uploaded PDFs/DOC/DOCX/TXT into an in-memory Chroma DB
    • Pulls the most relevant passages to seed the LLM
    • Prompts GPT-4o-mini (or whatever model is set in secrets.toml)
    • Validates / writes CSV, returns a ready-to-download ZIP
    """
    # ------------------------ guard-rails ---------------------------------
    if n > MAX_PATIENTS:
        raise ValueError(f"n={n} exceeds MAX_PATIENTS={MAX_PATIENTS}")
    if not (icd_code and icd_label):
        raise ValueError("ICD-10 code/label missing")

    if seed:
        random.seed(seed)

    # ------------------------ build prompt pieces -------------------------
    prompt_parts = [
        "You are a clinical data engine that fabricates HIGH-QUALITY "
        "synthetic patients strictly for software testing and demo purposes.",
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

    # ingest documents → Chroma -------------------------------------------
    notes      = [ln.strip() for ln in comments.splitlines()]
    vectordb   = ingest(files, notes) if files else None

    # pull top passages ----------------------------------------------------
    if vectordb:
        query    = f"{icd_label} clinical features comorbidities treatment"
        top_docs = vectordb.similarity_search(query, k=6)
        contexts = "\n---\n".join(
            d.page_content.strip()[:1_400] for d in top_docs
        )

        prompt_parts.append(
            "Use the following reference snippets ONLY for clinical realism. "
            "Never copy text verbatim:\n" + contexts
        )

    # require strict JSON output ------------------------------------------
    prompt_parts.append(
        textwrap.dedent(
            """
            Respond ONLY with valid JSON – an array of objects.
            Each object must contain:

            - patient_id   (string, unique)
            - age          (integer)
            - sex          ("F" or "M")
            - race         (US CDC wide-band)
            - ethnicity    ("Hispanic or Latino" / "Not Hispanic or Latino")
            - icd10_code   (string)
            - diagnosis    (string)

            Return NO markdown fences, NO commentary – pure JSON.
            """
        )
    )

    system_msg = {"role": "system",
                  "content": "You are a careful medical data generator."}
    user_msg   = {"role": "user", "content": "\n".join(prompt_parts)}

    # ------------------------ LLM call ------------------------------------
    response  = chat([system_msg, user_msg])
    raw_json  = response.choices[0].message.content.strip()

    try:
        patients = json.loads(raw_json)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"LLM returned invalid JSON: {err}") from None

    # ------------------------ CSV output ----------------------------------
    tmp_dir  = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / "patients.csv"
    headers  = ["patient_id", "age", "sex", "race", "ethnicity",
                "icd10_code", "diagnosis"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in patients:
            writer.writerow(row)

    # ------------------------ ZIP bundle ----------------------------------
    zip_path = tmp_dir / "cohort.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, csv_path.name)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return zip_path, run_id

