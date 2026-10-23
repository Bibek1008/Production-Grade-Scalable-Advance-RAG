#!/usr/bin/env python3
"""Render ECS task definitions using values from .env and AWS state files."""

import re
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DEF_DIR = REPO_ROOT / ".aws" / "task-definitions"
OUTPUT_DIR = Path(tempfile.gettempdir())


def load_arns():
    arns = {}
    arns_file = REPO_ROOT / "scripts" / "aws_secret_arns.sh"
    if arns_file.exists():
        with arns_file.open() as f:
            for line in f:
                line = line.strip().removeprefix("export ")
                m = re.match(r'([A-Z_]+)_ARN="([^"]+)"', line)
                if m:
                    arns[f"{m.group(1)}_ARN"] = m.group(2)
    return arns


def load_state():
    state = {}
    state_file = REPO_ROOT / "scripts" / "aws_deploy_state.sh"
    if state_file.exists():
        with state_file.open() as f:
            for line in f:
                # Handle both 'export KEY=value' and 'KEY=value' formats.
                line = line.strip().removeprefix("export ")
                m = re.match(r"([A-Z_0-9]+)=(.+)", line)
                if m:
                    state[m.group(1)] = m.group(2)
    return state


def render_task_definitions():
    arns = load_arns()
    state = load_state()

    image_uri = f"{state['ECR_URI']}:latest"
    backend_url = f"http://{state['ALB_DNS']}"
    placeholders = {
        "<IMAGE_NAME>": image_uri,
        "<AWS_REGION>": state.get("AWS_REGION", "us-east-1"),
        "<BACKEND_URL>": backend_url,
    }
    # The task definition templates use <ARN_NAME> placeholders for secrets.
    for key, value in arns.items():
        placeholders[f"<{key}>"] = value

    for family in ("rag-api", "rag-ui"):
        template_path = TASK_DEF_DIR / f"{family}.json"
        output_path = OUTPUT_DIR / f"{family}.json"
        with template_path.open() as f:
            content = f.read()
        for placeholder, value in placeholders.items():
            content = content.replace(placeholder, value)
        with output_path.open("w") as f:
            f.write(content)
        print(f"Rendered {output_path}")


if __name__ == "__main__":
    render_task_definitions()
