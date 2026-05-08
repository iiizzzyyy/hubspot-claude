from hubspot_agent.agents.cms import get_cms_agent_prompt


def test_cms_agent_prompt_has_correct_tools():
    prompt = get_cms_agent_prompt()
    assert prompt.agent_name == "CMS Agent"
    expected = [
        "hubspot_get_page",
        "hubspot_update_page",
        "hubspot_list_files",
        "hubspot_upload_file",
        "hubspot_publish_social_post",
    ]
    assert sorted(prompt.tool_names) == sorted(expected)
    assert "page" in prompt.system_prompt
    assert "file" in prompt.system_prompt
    assert "social" in prompt.system_prompt
