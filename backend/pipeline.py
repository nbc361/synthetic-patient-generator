# backend/pipeline.py
"""End-to-end cohort generator."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import tempfile, zipfile, csv, json, textwrap, random

# ── third-party -----------------------------------------------------------
from pydantic import BaseModel, ValidationError, field_validator

# ── local helpers ---------------------------------------------------------
from backend.openai_utils  import chat, MAX_PATIENTS          # OpenAI wrapper
from backend.data_ingest   import ingest                      # document loader


# ════════════════════════════════════════════════════════════════════════
# 1 ▪ Row schema & validator  ────────────────────────────────────────────
#    (kept tiny & fast — no external libs except Pydantic)
# ════════════════════════════════════════════════════════════════════════
class PatientRow(BaseModel):
    patient_id : str
    age        : int
    sex        : str
    race       : str
    ethnicity  : str
    icd10_code : str
    diagnosis  : str

    # ── coercion & sanity checks ────────────────────────────────────────
    @field_validator("age")
    @classmethod
    def _age_ok(cls, v: int) -> int:
        if not 0 <= v <= 120:
            raise ValueError("age must be 0–120")
        return v

    @field_validator("sex")
    @classmethod
    def _sex_ok(cls, v: str) -> str:
        v = v.upper()
        if v not in {"F", "M"}:
            raise ValueError("sex must be 'F' or 'M'")
        return v

    @field_validator("ethnicity")
    @classmethod
    def _eth_ok(cls, v: str) -> str:
        allowed = {
            "HISPANIC OR LATINO",
            "NOT HISPANIC OR LATINO",
        }
        if v.upper() not in allowed:
            raise ValueError("ethnicity must follow CDC wording")
        return v.title()

    @field_validator("icd10_code", mode="before")
    @classmethod
    def _code_upper(cls, v: str) -> str:
        return v.upper().strip()


# ════════════════════════════════════════════════════════════════════════
# 2 ▪ Main entry-point used by app.py
# ════════════════════════════════════════════════════════════════════════
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
    • Validates / coerces rows, writes CSV, returns a ready ZIP
    """

    # ── guard-rails ─────────────────────────────────────────────────────
    if n > MAX_PATIENTS:
        raise ValueError(f"n={n} exceeds MAX_PATIENTS={MAX_PATIENTS}")
    if not (icd_code and icd_label):
        raise ValueError("ICD-10 code/label missing")

    if seed:
        random.seed(seed)

    # ── prompt assembly ────────────────────────────────────────────────
    prompt_parts = [
        "You are a clinical data engine that fabricates HIGH-QUALITY "
        "synthetic patients strictly for software testing and demo purposes.",
        "",
        f"Diagnosis to model: **{icd_label}** (ICD-10 {icd_code}).",
        f"Number of patients requested: **{n}**.",
    ]

    # demographic filters ------------------------------------------------
    if demo_filters:
        readable = ", ".join(
            f"{k.replace('_',' ')}={v}" for k, v in demo_filters.items() if v
        )
        if readable:
            prompt_parts.append(f"Apply these demographic constraints: {readable}")

    # ingest reference material -----------------------------------------
    notes      = [ln.strip() for ln in comments.splitlines()]
    vectordb   = ingest(files, notes) if files else None

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

    # strict JSON output request ----------------------------------------
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

    # ── LLM call ────────────────────────────────────────────────────────
    response  = chat([system_msg, user_msg])
    raw_json  = response.choices[0].message.content.strip()

    try:
        raw_rows = json.loads(raw_json)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"LLM returned invalid JSON: {err}") from None

    # -------------------------------------------------------------------
    # ❷ Validate & coerce rows here
    # -------------------------------------------------------------------
    rows: list[PatientRow] = []
    seen_ids: set[str] = set()

    for idx, r in enumerate(raw_rows, 1):
        try:
            patient = PatientRow.model_validate(r)
        except ValidationError as e:
            raise RuntimeError(f"Row {idx} failed validation →\n{e}") from None

        if patient.icd10_code != icd_code.upper():
            raise RuntimeError(
                f"Row {idx}: icd10_code '{patient.icd10_code}' ≠ requested '{icd_code}'"
            )
        if patient.patient_id in seen_ids:
            raise RuntimeError(f"Duplicate patient_id '{patient.patient_id}'")
        seen_ids.add(patient.patient_id)
        rows.append(patient)

    if len(rows) != n:
        raise RuntimeError(f"Model returned {len(rows)} rows, expected {n}")

    # ── CSV output ──────────────────────────────────────────────────────
    tmp_dir  = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / "patients.csv"
    headers  = PatientRow.model_fields.keys()

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for p in rows:
            writer.writerow([getattr(p, h) for h in headers])

    # ── ZIP bundle ──────────────────────────────────────────────────────
    zip_path = tmp_dir / "cohort.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, csv_path.name)

    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return zip_path, run_id
