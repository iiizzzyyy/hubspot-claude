from hubspot_agent.orchestrator import reconcile_after_timeout


def test_reconcile_after_timeout():
    result = reconcile_after_timeout(
        portal_id="123",
        expected_action="bulk_update_contacts",
        expected_payload={"inputs": [{"id": "1", "properties": {"email": "a@b.com"}}]},
    )
    assert result["portal_id"] == "123"
    assert result["expected_action"] == "bulk_update_contacts"
    assert result["reconciliation_needed"] is True
    assert "HygieneAgent" in result["instruction"]
    assert "action_id" in result
