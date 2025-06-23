# backend/pipeline.py
"""End-to-end synthetic-cohort generator (dynamic extra columns supported)."""

from __future__ import annotations

# ── std-lib ───────────────────────────────────────────────────────────────
import csv, json, os, random, tempfile, textwrap, zipfile
from datetime import datetime
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────
from pydantic import BaseModel, ValidationError, field_validator

# ── local helpers ─────────────────────────────────────────────────────────
from backend.openai_utils import chat, MAX_PATIENTS, MODEL, TEMPERATURE
from backend.data_ingest  import ingest                          # PDF/DOC/TXT ⇒ Chroma

# ════════════════════════════════════════════════════════════════════════
# 0 ▪ helper for textarea schema
# ════════════════════════════════════════════════════════════════════════
def _parse_extra_schema(text: str) -> list[tuple[str, str]]:
    """Turn textarea lines into [(field, type)]  — types = int/float/str."""
    cols: list[tuple[str, str]] = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        if ":" not in ln:
            raise ValueError(f"Missing ':' in line → {ln!r}")
        name, typ = [p.strip() for p in ln.split(":", 1)]
        if not name.isidentifier():
            raise ValueError(f"Invalid field name → {name!r}")
        typ = typ.lower()
        if typ not in {"int", "float", "str"}:
            raise ValueError(f"Type must be int / float / str → {typ!r}")
        cols.append((name, typ))
    if not cols:
        raise ValueError("No columns recognised")
    return cols


# ════════════════════════════════════════════════════════════════════════
# 1 ▪ core patient schema
# ════════════════════════════════════════════════════════════════════════
class PatientRow(BaseModel):
    patient_id: str
    age:        int
    sex:        str
    race:       str
    ethnicity:  str
    icd10_code: str
    diagnosis:  str

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
        allowed = {"HISPANIC OR LATINO", "NOT HISPANIC OR LATINO"}
        if v.upper() not in allowed:
            raise ValueError("ethnicity must follow CDC wording")
        return v.title()

    @field_validator("icd10_code", mode="before")
    @classmethod
    def _code_upper(cls, v: str) -> str:
        return v.upper().strip()


# ════════════════════════════════════════════════════════════════════════
# 2 ▪ public entry-point
# ════════════════════════════════════════════════════════════════════════
def generate_cohort(
    *,
    icd_code:      str,
    icd_label:     str,
    files,                      # list[streamlit.UploadedFile]
    comments:      str,
    n:             int,
    demo_filters:  dict,
    seed:          str | None = None,
    benchmark                   = None,
    extra_schema:  str = "",          # ← textarea from the UI
):
    # ── guard rails ────────────────────────────────────────────────────
    if n > MAX_PATIENTS:
        raise ValueError(f"n={n} exceeds MAX_PATIENTS={MAX_PATIENTS}")
    if not (icd_code and icd_label):
        raise ValueError("ICD-10 code/label missing")
    if seed:
        random.seed(seed)

    # ── parse extra columns — at most 10 ───────────────────────────────
    extra_cols: list[tuple[str, str]] = []
    if extra_schema.strip():
        extra_cols = _parse_extra_schema(extra_schema)
        if len(extra_cols) > 10:
            raise ValueError("Max 10 extra columns allowed")

    # ── prompt assembly ────────────────────────────────────────────────
    prompt_parts = [
        "You are a clinical data engine that fabricates HIGH-QUALITY "
        "synthetic patients strictly for software testing and demo purposes.",
        f"Diagnosis to model: **{icd_label}** (ICD-10 {icd_code}).",
        f"Number of patients requested: **{n}**.",
    ]

    if demo_filters:
        df_readable = ", ".join(f"{k.replace('_',' ')}={v}"
                                for k, v in demo_filters.items() if v)
        if df_readable:
            prompt_parts.append(f"Apply these demographic constraints: {df_readable}")

    if extra_cols:
        want = ", ".join(f"{nm} ({tp})" for nm, tp in extra_cols)
        prompt_parts.append(
            f"Additionally include these attributes on each patient: {want}"
        )

    # ── context from reference PDFs/DOCX/TXT ──────────────────────────
    notes = [ln.strip() for ln in comments.splitlines()]
    vectordb = ingest(files, notes) if files else None
    if vectordb:
        q   = f"{icd_label} clinical features comorbidities treatment epidemiology"
        ctx = vectordb.similarity_search(q, k=6)
        snippets = "\n---\n".join(d.page_content.strip()[:1_400] for d in ctx)
        prompt_parts.append(
            "Use the following snippets ONLY for clinical realism. "
            "Never copy text verbatim:\n" + snippets
        )

    # ── strict JSON spec ───────────────────────────────────────────────
    core_block  = "\n".join(f"- {f}" for f in PatientRow.model_fields.keys())
    extra_block = "\n".join(f"- {nm} ({tp})" for nm, tp in extra_cols)
    prompt_parts.append(
        textwrap.dedent(
            f"""
            Respond ONLY with valid JSON – an array of objects.
            Each object must contain **all** of:

{core_block}
{extra_block or ''}
            Return NO markdown fences, NO commentary – pure JSON.
            """
        )
    )

    # ── LLM call ───────────────────────────────────────────────────────
    msgs = [
        {"role": "system", "content": "You are a careful medical data generator."},
        {"role": "user",   "content": "\n".join(prompt_parts)},
    ]
    response = chat(msgs)
    raw_json = response.choices[0].message.content.strip()

    try:
        raw_rows = json.loads(raw_json)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"LLM returned invalid JSON: {err}") from None

    # ── validate rows ---------------------------------------------------
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for idx, r in enumerate(raw_rows, 1):
        try:
            core = PatientRow.model_validate(r)           # type: ignore[arg-type]
        except ValidationError as e:
            raise RuntimeError(f"Row {idx} failed core validation →\n{e}") from None

        if core.icd10_code != icd_code.upper():
            raise RuntimeError(
                f"Row {idx}: icd10_code '{core.icd10_code}' ≠ requested '{icd_code}'"
            )
        if core.patient_id in seen_ids:
            raise RuntimeError(f"Duplicate patient_id '{core.patient_id}'")
        seen_ids.add(core.patient_id)

        # type-check extras
        for col, typ in extra_cols:
            if col not in r:
                raise RuntimeError(f"Row {idx} missing extra column '{col}'")
            if typ == "int"   and not isinstance(r[col], int):
                raise RuntimeError(f"Row {idx}: {col} must be int")
            if typ == "float" and not isinstance(r[col], (int, float)):
                raise RuntimeError(f"Row {idx}: {col} must be float")
            if typ == "str"   and not isinstance(r[col], str):
                raise RuntimeError(f"Row {idx}: {col} must be str")

        rows.append(r)

    if len(rows) != n:
        raise RuntimeError(f"Model returned {len(rows)} rows, expected {n}")

    # ── CSV output -------------------------------------------------------
    tmp_dir  = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / "patients.csv"
    headers  = list(PatientRow.model_fields.keys()) + [nm for nm, _ in extra_cols]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})

    # ── metadata ---------------------------------------------------------
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    meta = {
        "run_id": run_id,
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "icd10_code": icd_code,
        "icd10_label": icd_label,
        "n_requested": n,
        "demographics": demo_filters,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed": seed,
        "extra_columns": {nm: tp for nm, tp in extra_cols},
    }
    meta_path = tmp_dir / "run_meta.json"
    json.dump(meta, meta_path.open("w"), indent=2)

    # ── ZIP bundle -------------------------------------------------------
    zip_path = tmp_dir / "cohort.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path,  csv_path.name)
        z.write(meta_path, meta_path.name)

    return zip_path, run_id
