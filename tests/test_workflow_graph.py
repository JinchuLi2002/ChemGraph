"""Tests for the Streamlit workflow graph model."""

from ui.workflow_graph import build_workflow_graph, _workflow_graph_html


def test_build_workflow_graph_starts_with_query_only():
    graph = build_workflow_graph("Optimize water.", [])

    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["kind"] == "query"
    assert graph["nodes"][0]["output"] == "Optimize water."


def test_build_workflow_graph_exposes_tool_input_and_output():
    messages = [
        {"type": "human", "content": "Calculate the energy of water."},
        {
            "type": "ai",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "run_pyscf",
                    "args": {"molecule": "water", "basis": "sto-3g"},
                }
            ],
        },
        {
            "type": "tool",
            "name": "run_pyscf",
            "tool_call_id": "call-1",
            "content": '{"energy_hartree": -75.0}',
        },
        {"type": "ai", "content": "The energy is -75.0 Hartree."},
    ]

    graph = build_workflow_graph("Calculate the energy of water.", messages)

    tool_nodes = [node for node in graph["nodes"] if node["kind"] == "tool"]
    assert len(tool_nodes) == 1
    assert tool_nodes[0]["title"] == "run_pyscf"
    assert '"basis": "sto-3g"' in tool_nodes[0]["input"]
    assert "-75.0" in tool_nodes[0]["output"]

    final_assistant = graph["nodes"][-1]
    assert final_assistant["kind"] == "assistant"
    assert "run_pyscf output" in final_assistant["input"]
    assert "The energy is -75.0 Hartree." in final_assistant["output"]


def test_build_workflow_graph_handles_openai_raw_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "additional_kwargs": {
                "tool_calls": [
                    {
                        "id": "raw-call-1",
                        "type": "function",
                        "function": {
                            "name": "molecule_name_to_smiles",
                            "arguments": '{"name": "caffeine"}',
                        },
                    }
                ]
            },
        },
        {
            "role": "tool",
            "name": "molecule_name_to_smiles",
            "tool_call_id": "raw-call-1",
            "content": '{"smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"}',
        },
    ]

    graph = build_workflow_graph("Find caffeine smiles.", messages)

    tool_node = next(node for node in graph["nodes"] if node["kind"] == "tool")
    assert tool_node["title"] == "molecule_name_to_smiles"
    assert '"name": "caffeine"' in tool_node["input"]
    assert "Cn1cnc2c1c(=O)n(C)c(=O)n2C" in tool_node["output"]


def test_workflow_graph_html_includes_pan_zoom_controls():
    graph = build_workflow_graph("Optimize water.", [])
    html = _workflow_graph_html(graph, key="test_graph")

    assert 'data-zoom="in"' in html
    assert 'data-zoom="out"' in html
    assert 'data-zoom="fit"' in html
    assert "pointerdown" in html
    assert "wheel" in html
