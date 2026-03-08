import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "slimmed_experts"
PYTHON_VERSION = "3.12"

# Project commands
@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run zensical build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run zensical serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

@task
def clean(ctx: Context) -> None:
    """Clean up build artifacts."""
    ctx.run("rm -rf build .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info", echo=True, pty=not WINDOWS)
    ctx.run("find . -type f -name '*.py[co]' -delete", echo=True, pty=not WINDOWS)
    ctx.run("find . -type d -name '__pycache__' -delete", echo=True, pty=not WINDOWS)
