"""API smoke tests over the full serving path (data + model seeded by conftest)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"] is not None  # model was trained


def test_forecast_endpoint():
    r = client.get("/forecast/1?horizon=7")
    assert r.status_code == 200
    body = r.json()
    assert body["item_id"] == 1
    assert len(body["forecast"]) == 7
    assert all(pt["units"] >= 0 for pt in body["forecast"])


def test_recommendations_endpoint():
    r = client.get("/recommendations")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    first = body["recommendations"][0]
    for key in ("recommended_order", "spoilage_risk_qty", "money_at_risk"):
        assert key in first


def test_spoilage_risk_endpoint():
    r = client.get("/spoilage-risk")
    assert r.status_code == 200
    items = r.json()["items"]
    # all returned items must actually have money at risk, ranked descending
    assert all(i["money_at_risk"] > 0 for i in items)
    risks = [i["money_at_risk"] for i in items]
    assert risks == sorted(risks, reverse=True)
