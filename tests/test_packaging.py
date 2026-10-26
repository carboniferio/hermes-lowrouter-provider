"""What gets installed, what the docs page renders, what the code is allowed to touch."""

from __future__ import annotations

import ast
from pathlib import Path

from conftest import PLUGIN_DIR, ROOT, plugin

SOURCES = sorted(PLUGIN_DIR.glob("*.py"))


def _manifest() -> dict:
    fields = {}
    lines = (PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line and not line.startswith(" ") and ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    fields["requires_env"] = [line.strip()[2:] for line in lines if line.startswith("  - ")]
    return fields


def _imports(path: Path) -> list:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_the_plugin_is_one_file():
    assert [path.name for path in SOURCES] == ["__init__.py"]


def test_manifest_declares_a_model_provider():
    manifest = _manifest()
    assert manifest["name"] == "lowrouter"
    assert manifest["kind"] == "model-provider"
    assert manifest["requires_env"] == [plugin.API_KEY_ENV]


def test_plugin_readme_matches_repo_readme():
    # docs page renders lowrouter/README.md, GitHub the root one
    assert (PLUGIN_DIR / "README.md").read_bytes() == (ROOT / "README.md").read_bytes()


def test_readme_links_are_absolute():
    # README is rendered off-repo, relative links break
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for part in readme.split("](")[1:]:
        assert part.startswith("https://"), part[:60]


def test_manifest_version_is_the_one_the_readme_documents():
    assert f"### {_manifest()['version']}" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_only_the_standard_library_and_hermes_are_imported():
    allowed = {"__future__", "datetime", "hashlib", "json", "logging", "threading", "time", "typing", "urllib.request",
               "providers", "providers.base", "hermes_cli.urllib_security", "hermes_cli.auth",
               "agent.account_usage"}
    for path in SOURCES:
        assert set(_imports(path)) <= allowed, path.name


def test_network_goes_through_hermes_and_nothing_else_is_touched():
    banned = {"open", "exec", "eval", "compile", "__import__", "system", "popen", "getenv", "urlopen",
              "Thread", "write_text", "write_bytes", "read_text", "read_bytes", "mkdir", "remove", "unlink"}
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
                assert called not in banned, f"{path.name}:{node.lineno} calls {called}()"
            if isinstance(node, ast.Attribute):
                assert node.attr != "environ", f"{path.name}:{node.lineno} reads os.environ"


def test_registers_exactly_one_provider():
    tree = ast.parse((PLUGIN_DIR / "__init__.py").read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("register")]
    assert [call.func.id for call in calls] == ["register_provider"]
