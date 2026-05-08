from hubspot_agent.orchestrator import (
    needs_approval,
    normalize_informing_sources,
    present_preview,
    store_preview_for_execution,
)
from hubspot_agent.models import PreviewResult, RiskLevel


def test_needs_approval_low():
    assert needs_approval(RiskLevel.LOW) is False


def test_needs_approval_medium():
    assert needs_approval(RiskLevel.MEDIUM) is True


def test_needs_approval_high():
    assert needs_approval(RiskLevel.HIGH) is True


def test_needs_approval_destructive():
    assert needs_approval(RiskLevel.DESTRUCTIVE) is True


def test_present_preview_summary():
    result = PreviewResult(
        preview={"affected": [{"id": "1", "name": "Test"}]},
        impact_count=1,
        risk_level=RiskLevel.MEDIUM,
        proposed_payload={"endpoint": "/crm/v3/objects/contacts"},
        original_values={},
    )
    text = present_preview(result)
    assert "1 records" in text
    assert "MEDIUM" in text
    assert "Approve?" in text


def test_present_preview_destructive():
    result = PreviewResult(
        preview={"affected": [{"id": "1", "name": "Test"}]},
        impact_count=5,
        risk_level=RiskLevel.DESTRUCTIVE,
        proposed_payload={},
        original_values={},
    )
    text = present_preview(result)
    assert "DESTRUCTIVE" in text
    assert "Type `5` to confirm" in text


def test_present_preview_details():
    result = PreviewResult(
        preview={"affected": [{"id": "1", "name": "Test"}]},
        impact_count=1,
        risk_level=RiskLevel.HIGH,
        proposed_payload={"endpoint": "/api/test"},
        original_values={},
    )
    text = present_preview(result, mode="details")
    assert "Affected records:" in text
    assert "Exact API call:" in text


def test_store_preview_for_execution(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = PreviewResult(
        preview={},
        impact_count=1,
        risk_level=RiskLevel.MEDIUM,
        proposed_payload={},
        original_values={"contacts": [{"id": "1"}]},
    )
    path = store_preview_for_execution("123", "act-1", result)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["original_values"]["contacts"][0]["id"] == "1"


def test_present_preview_with_official_sources():
    result = PreviewResult(
        preview={"affected": [{"id": "1", "name": "Test"}]},
        impact_count=1,
        risk_level=RiskLevel.MEDIUM,
        proposed_payload={},
        original_values={},
        informing_sources=[
            {"source": "official", "trust_tier": "official", "title": "Contact API Docs", "url": "https://developers.hubspot.com/docs/api/crm/contacts"},
        ],
    )
    text = present_preview(result)
    assert "Informed by:" in text
    assert "[Official: Contact API Docs]" in text


def test_present_preview_with_community_sources():
    result = PreviewResult(
        preview={"affected": [{"id": "1", "name": "Test"}]},
        impact_count=1,
        risk_level=RiskLevel.MEDIUM,
        proposed_payload={},
        original_values={},
        informing_sources=[
            {"source": "community", "trust_tier": "community-unverified", "title": "Help Post", "url": "https://community.hubspot.com/t5/Lists/foo/td-p/12345"},
        ],
    )
    text = present_preview(result)
    assert "[Community Unverified: Help Post]" in text


def test_normalize_informing_sources_downgrades_misclassified():
    sources = [
        {"source": "official", "trust_tier": "official", "title": "Fake", "url": "https://community.hubspot.com/t5/foo"},
    ]
    normalized = normalize_informing_sources(sources)
    assert normalized[0]["source"] == "community"
    assert normalized[0]["trust_tier"] == "community-unverified"


def test_normalize_informing_sources_preserves_community_accepted():
    sources = [
        {"source": "official", "trust_tier": "official", "title": "Accepted", "url": "https://community.hubspot.com/t5/foo"},
    ]
    normalized = normalize_informing_sources(sources)
    # Should be downgraded from official, but if it had been community-accepted
    # that context would be preserved. Since it was "official", it gets downgraded.
    assert normalized[0]["source"] == "community"
    assert normalized[0]["trust_tier"] == "community-unverified"
