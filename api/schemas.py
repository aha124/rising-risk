"""Request and response models.

The request model is built from artifacts/model/feature_spec.json when the service
starts, so the API accepts exactly the features the saved model was trained on.

Validation rules for a feature object:

- Unknown keys are rejected. A typo should fail, not be scored as missing.
- Any feature may be left out or sent as null. It is scored as missing, the way the
  trees were trained to handle it, and counted in `n_missing` in the response.
- 0/1 features must be 0 or 1. Whole-number features must be whole numbers.
- Numeric features have hard limits wider than the training data: the training range
  extended by half its width on each side, with a floor of 0 where the training
  minimum is 0 or more. That catches unit mistakes (cents for dollars, months for
  years) without rejecting a member who is a little outside what training saw.
- A value inside the hard limits but outside the training range is scored, and named
  in `warnings`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

MAX_BATCH = 1000


def hard_limits(spec: dict) -> tuple[float, float]:
    lo, hi = float(spec["min"]), float(spec["max"])
    if spec["type"] == "binary":
        return 0.0, 1.0
    pad = (hi - lo) / 2
    lower = 0.0 if lo >= 0 else lo - pad
    return lower, hi + pad


class _FeatureBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def _whole_numbers(self):
        for name in self._whole_number_fields():
            value = getattr(self, name)
            if value is not None and value != round(value):
                raise ValueError(f"{name} must be a whole number, got {value}")
        return self

    @classmethod
    def _whole_number_fields(cls) -> tuple[str, ...]:
        return ()


def build_feature_model(spec: list[dict]) -> type[BaseModel]:
    """A pydantic model with one optional numeric field per model feature."""
    fields = {}
    for f in spec:
        lower, upper = hard_limits(f)
        unit = f" Unit: {f['unit']}." if f.get("unit") else ""
        fields[f["name"]] = (
            float | None,
            Field(
                default=None,
                ge=lower,
                le=upper,
                title=f["label"],
                description=f"{f['description']}{unit}",
            ),
        )
    whole = tuple(f["name"] for f in spec if f.get("integer_valued") or f["type"] == "binary")
    model = create_model("Features", __base__=_FeatureBase, **fields)
    model._whole_number_fields = classmethod(lambda cls: whole)
    return model


class Factor(BaseModel):
    feature: str
    label: str
    value: str = Field(description="The input as a person would write it.")
    contribution: float = Field(description="TreeSHAP contribution to the score, in log-odds.")
    direction: Literal["raises", "lowers"]


class ScoreResponse(BaseModel):
    probability: float = Field(description="Calibrated probability of a top-decile cost year.")
    percentile: float | None = Field(
        description="Where the probability sits among test-year members, 0 to 100."
    )
    tier: Literal["high", "rising", "routine"]
    top_factors: list[Factor]
    n_missing: int = Field(description="Features left out or sent as null.")
    warnings: list[str]
    model_version: str
    request_id: str


class BatchItem(BaseModel):
    index: int
    probability: float
    percentile: float | None
    tier: Literal["high", "rising", "routine"]
    top_factors: list[Factor]
    n_missing: int
    warnings: list[str]


class BatchResponse(BaseModel):
    results: list[BatchItem]
    n: int
    model_version: str
    request_id: str


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    n_features: int
