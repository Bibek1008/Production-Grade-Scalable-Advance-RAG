"""Tests for source-grounded troubleshooting responses."""

from types import SimpleNamespace

import app.agents.nodes.responder as responder


def test_troubleshooting_prompt_uses_documented_facts_without_inventing_steps(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        responder,
        "_generate_response",
        lambda prompt: prompts.append(prompt) or SimpleNamespace(content="Source-grounded answer."),
    )
    state = {
        "current_query": "kubernetes troubleshoot networking",
        "messages": [{"role": "user", "content": "How do I troubleshoot Kubernetes networking?"}],
        "documents": [
            {
                "content": "All Pods can communicate without NAT. Networking uses a CNI plugin.",
                "source": "architecture.pptx",
                "score": 0.9,
            }
        ],
        "plan": ["Intent: Technical"],
        "status": "Found technical context.",
    }

    result = responder.generate_node(state)

    assert result["final_answer"] == "Source-grounded answer."
    assert "label these as checks" in prompts[0]
    assert "inventing them" in prompts[0]
    assert "Source: architecture.pptx" in prompts[0]
