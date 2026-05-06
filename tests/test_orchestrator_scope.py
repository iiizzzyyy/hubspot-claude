from hubspot_agent.orchestrator import validate_scopes


def test_validate_scopes_with_empty_agent_names():
    result = validate_scopes([], ["crm.objects.contacts.read"])
    assert result == {}


def test_validate_scopes_unknown_agent():
    result = validate_scopes(["nonexistent"], ["crm.objects.contacts.read"])
    assert result == {}
