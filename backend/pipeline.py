# backend/pipeline.py
"""End-to-end synthetic-cohort generator (dynamic columns, units, cost)."""

from __future__ import annotations

# ── std-lib ──────────────────────────────────────────────────────────────
import csv, json, random, tempfile, textwrap, zipfile
from datetime import datetime
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────
from pydantic import BaseModel, ValidationError, field_validator

# ── local helpers ────────────────────────────────────────────────────────
from backend.openai_utils import chat, MAX_PATIENTS, MODEL, TEMPERATURE
from backend.data_ingest  import ingest                       # PDF/DOC/TXT ⇒ Chroma

# ════════════════════════════════════════════════════════════════════════
# 0 ▪ textarea helper  –  now supports optional unit
# ════════════════════════════════════════════════════════════════════════
# ----------------------------------------------------------------------
def _parse_extra_schema(text: str) -> list[tuple[str, str]]:
    """
    Parse textarea lines into [(field_name, type), …]

    • Accepts exactly one “:” per non-blank line.
    • Allowed types → int | float | str  (case-insensitive).
    • Ignores leading bullets (✓, *, •) and surrounding whitespace.
    """
    cols: list[tuple[str, str]] = []

    for raw in text.splitlines():
        ln = raw.lstrip("•*✓- ").strip()          # strip any bullets / dashes
        if not ln:
            continue                              # skip blank lines

        if ln.count(":") != 1:
            raise ValueError(f"Expect one ':' → {raw!r}")

        name, typ = [p.strip() for p in ln.split(":", 1)]

        if not name.isidentifier():
            raise ValueError(f"Invalid field name → {name!r}")

        typ = typ.lower()
        if typ not in {"int", "float", "str"}:
            raise ValueError(f"Type must be int / float / str → {typ!r}")

        cols.append((name, typ))

    if not cols:
        raise ValueError("No columns recognised")

    return cols          # e.g. [("fev1_pct","float"), …]

# ════════════════════════════════════════════════════════════════════════
# 1 ▪ Core patient schema  (kept tiny & fast)
# ════════════════════════════════════════════════════════════════════════
class PatientRow(BaseModel):
    patient_id : str
    age        : int
    sex        : str
    race       : str
    ethnicity  : str
    icd10_code : str
    diagnosis  : str

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
    def _upper(cls, v: str) -> str:
        return v.upper().strip()


# ════════════════════════════════════════════════════════════════════════
# 2 ▪ Public API  (called by Streamlit front-end)
# ════════════════════════════════════════════════════════════════════════
def generate_cohort(
    *,
    icd_code:      str,
    icd_label:     str,
    files,
    comments:      str,
    n:             int,
    demo_filters:  dict,
    seed:          str | None = None,
    benchmark                   = None,
    extra_schema:   str         = "",
):
    """Return (zip_path, run_id)."""

    # ── guard rails ────────────────────────────────────────────────────
    if n > MAX_PATIENTS:
        raise ValueError(f"n={n} > MAX_PATIENTS={MAX_PATIENTS}")
    if not (icd_code and icd_label):
        raise ValueError("ICD-10 code/label missing")
    if seed:
        random.seed(seed)

    # ── dynamic columns -------------------------------------------------
    extra_cols: list[tuple[str, str, str]] = []
    if extra_schema.strip():
        extra_cols = _parse_extra_schema(extra_schema)
        if len(extra_cols) > 10:
            raise ValueError("Max 10 extra columns")

    # ── prompt assembly ────────────────────────────────────────────────
    prompt_parts = [
        "You are a clinical data engine that fabricates HIGH-QUALITY synthetic "
        "patients strictly for software testing and demo. Follow ALL rules:",
        "— Do NOT invent lab tests or units that do not exist.",
        "— Use realistic human ranges (e.g. weight 40-200 kg; FEV1 10-110 %).",
        "— NEVER copy text verbatim from reference passages.",
        "",
        f"Diagnosis to model: **{icd_label}** (ICD-10 {icd_code}).",
        f"Number of patients requested: **{n}**.",
    ]

    if demo_filters:
        readable = ", ".join(
            f"{k.replace('_',' ')}={v}" for k, v in demo_filters.items() if v
        )
        if readable:
            prompt_parts.append(f"Apply these demographic constraints: {readable}")

    if extra_cols:
        prompt_parts.append(
            "Additionally include these attributes:"
            + ", ".join(f"{nm} ({tp})" for nm, tp, _ in extra_cols)
        )

    # ── context from reference documents --------------------------------
    notes    = [ln.strip() for ln in comments.splitlines()]
    vectordb = ingest(files, notes) if files else None
    if vectordb:
        q   = f"{icd_label} clinical features comorbidities treatment"
        ctx = vectordb.similarity_search(q, k=6)
        snippet = "\n---\n".join(d.page_content.strip()[:1_400] for d in ctx)
        prompt_parts.append(
            "Use the following snippets ONLY for realism; never copy:\n" + snippet
        )

    # strict JSON instruction -------------------------------------------
    core_fields   = "\n".join(f"- {f}" for f in PatientRow.model_fields.keys())
    extra_fields  = "\n".join(f"- {nm} ({tp})" for nm, tp, _ in extra_cols)
    prompt_parts.append(
        textwrap.dedent(
            f"""
            Respond ONLY with valid JSON – an array of objects.
            Each object must contain:

{core_fields}
{extra_fields or ''}

            NO markdown fences, NO commentary – pure JSON only.
            """
        )
    )

    # ── call OpenAI ─────────────────────────────────────────────────────
    msgs      = [
        {"role": "system", "content": "You are a careful medical data generator."},
        {"role": "user",   "content": "\n".join(prompt_parts)},
    ]
    response  = chat(msgs)                 # openai_utils adds _cost_usd
    raw_json  = response.choices[0].message.content.strip()

    try:
        raw_rows = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON → {e}") from None

    # ── validate rows ---------------------------------------------------
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for idx, r in enumerate(raw_rows, 1):

        # core checks
        try:
            core = PatientRow.model_validate(r)               # type: ignore[arg-type]
        except ValidationError as e:
            raise RuntimeError(f"Row {idx} core validation failed →\n{e}") from None

        if core.icd10_code != icd_code.upper():
            raise RuntimeError(f"Row {idx}: icd10_code mismatch")
        if core.patient_id in seen_ids:
            raise RuntimeError(f"Duplicate patient_id '{core.patient_id}'")
        seen_ids.add(core.patient_id)

        # extra cols checks
        for col, typ, _unit in extra_cols:
            if col not in r:
                raise RuntimeError(f"Row {idx} missing '{col}'")
            if typ == "int"   and not isinstance(r[col], int):
                raise RuntimeError(f"{col} must be int  (row {idx})")
            if typ == "float" and not isinstance(r[col], (int, float)):
                raise RuntimeError(f"{col} must be float  (row {idx})")
            if typ == "str"   and not isinstance(r[col], str):
                raise RuntimeError(f"{col} must be str  (row {idx})")

        rows.append(r)

    if len(rows) != n:
        raise RuntimeError(f"Model returned {len(rows)} rows, expected {n}")

    # ── CSV output ------------------------------------------------------
    tmp_dir  = Path(tempfile.mkdtemp())
    csv_path = tmp_dir / "patients.csv"
    headers  = list(PatientRow.model_fields.keys()) + [nm for nm, _, _ in extra_cols]

    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=headers)
        wr.writeheader()
        for r in rows:
            wr.writerow({h: r.get(h, "") for h in headers})

    # ── meta JSON -------------------------------------------------------
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    meta_path = tmp_dir / "run_meta.json"
    json.dump(
        {
            "run_id"          : run_id,
            "generated_utc"   : datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "icd10_code"      : icd_code,
            "icd10_label"     : icd_label,
            "n_requested"     : n,
            "demographics"    : demo_filters,
            "model"           : MODEL,
            "temperature"     : TEMPERATURE,
            "seed"            : seed,
            "extra_units"     : {nm: unit for nm, _, unit in extra_cols},
            "usage_tokens"    : {
                "prompt"     : getattr(response.usage, "prompt_tokens",     None),
                "completion" : getattr(response.usage, "completion_tokens", None),
                "total"      : getattr(response.usage, "total_tokens",      None),
            },
            "estimated_cost_usd": getattr(response, "_cost_usd", None),
        },
        meta_path.open("w"),
        indent=2,
    )

    # ── ZIP bundle ------------------------------------------------------
    zip_path = tmp_dir / "cohort.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path,  csv_path.name)
        zf.write(meta_path, meta_path.name)

    return zip_path, run_id
