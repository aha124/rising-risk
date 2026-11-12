"""A test client on the committed model, with the audit log in a temporary folder."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from riskmodel import scoring


@pytest.fixture(scope="session")
def scored():
    """The whole mart scored the way batch evaluation scores it."""
    return scoring.load_scored()


@pytest.fixture()
def audit_path(tmp_path):
    return tmp_path / "audit" / "audit.db"


@pytest.fixture()
def client(audit_path):
    with TestClient(create_app(audit_path=audit_path)) as c:
        yield c


def to_payload(features: list[str], row) -> dict:
    """A feature object for one row of the feature matrix. Missing values are left out."""
    return {name: float(v) for name, v in zip(features, row, strict=True) if not np.isnan(v)}


@pytest.fixture(scope="session")
def sample(scored):
    """600 member-years from every index year: the feature matrix and batch results."""
    rows = scored.frame.sample(600, random_state=11)
    X = scored.X(rows)
    assert np.isnan(X).any(), "the sample should include rows with missing values"
    payloads = [to_payload(scored.features, x) for x in X]
    return rows, X, payloads
