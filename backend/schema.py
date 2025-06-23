from pydantic import BaseModel, Field, validator
from typing import Literal

class Patient(BaseModel):
    patient_id : str                       = Field(..., min_length=2, max_length=40)
    age        : int                       = Field(..., ge=0, le=120)
    sex        : Literal["F", "M"]
    race       : str
    ethnicity  : Literal[
        "Hispanic or Latino",
        "Not Hispanic or Latino"
    ]
    icd10_code : str
    diagnosis  : str

    # ——— derived / sanity checks ———
    @validator("icd10_code")
    def upper_strip(cls, v):          # J47, L73.2, etc.
        return v.strip().upper()
