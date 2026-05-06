from hubspot_agent.orchestrator import dispatch_agent


def test_dispatch_agent_unknown():
    result = dispatch_agent("nonexistent", "do something")
    assert result.status == "error"
    assert "Unknown agent" in result.error_message


def test_dispatch_agent_preview_mode():
    result = dispatch_agent("objects", "find contacts in northeast")
    assert result.status == "preview"
    assert "find contacts in northeast" in result.data["full_prompt"]
    assert "Mode: preview" in result.data["full_prompt"]


def test_dispatch_agent_execute_mode():
    result = dispatch_agent(
        "objects",
        "create a contact",
        mode="execute",
        payload={"properties": {"email": "test@example.com"}},
    )
    assert result.status == "ready"
    assert "Execute the following payload" in result.data["full_prompt"]
    assert "test@example.com" in result.data["full_prompt"]
