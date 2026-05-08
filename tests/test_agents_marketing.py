from hubspot_agent.agents.marketing import get_marketing_agent_prompt


def test_marketing_agent_prompt_has_correct_tools():
    prompt = get_marketing_agent_prompt()
    assert prompt.agent_name == "Marketing Agent"
    expected = [
        "hubspot_create_email",
        "hubspot_get_email",
        "hubspot_create_campaign",
        "hubspot_list_campaigns",
        "hubspot_create_segment",
        "hubspot_get_segment",
        "hubspot_create_ab_test",
        "hubspot_get_ab_test",
        "hubspot_create_suppression_list",
        "hubspot_list_suppression_lists",
        "hubspot_send_email",
        "hubspot_get_email_performance",
    ]
    assert sorted(prompt.tool_names) == sorted(expected)
