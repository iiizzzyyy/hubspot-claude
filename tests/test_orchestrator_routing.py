from hubspot_agent.orchestrator import route_request


def test_route_objects_keywords():
    result = route_request("find contacts in northeast")
    assert "objects" in result


def test_route_properties_keywords():
    result = route_request("create a custom field for deals")
    assert "properties" in result


def test_route_workflows_keywords():
    result = route_request("build an automation workflow")
    assert "workflows" in result


def test_route_lists_keywords():
    result = route_request("add contacts to a list")
    assert "lists" in result


def test_route_pipelines_keywords():
    result = route_request("reorder deal pipeline stages")
    assert "pipelines" in result


def test_route_users_keywords():
    result = route_request("onboard a new user")
    assert "users" in result


def test_route_hygiene_keywords():
    result = route_request("find duplicate contacts")
    assert "hygiene" in result


def test_route_analytics_keywords():
    result = route_request("how many deals closed this month")
    assert "analytics" in result


def test_route_associations_keywords():
    result = route_request("link contacts to companies")
    assert "associations" in result


def test_route_engagements_keywords():
    result = route_request("log a call with the prospect")
    assert "engagements" in result


def test_route_raw_api_keywords():
    result = route_request("use raw api for custom endpoint")
    assert "raw_api" in result


def test_route_ambiguous_returns_empty():
    result = route_request("hello world")
    assert result == []


def test_route_multi_agent_dependency_order():
    result = route_request("create a property and then build a workflow")
    # properties should come before workflows due to dependency
    assert result.index("properties") < result.index("workflows")
