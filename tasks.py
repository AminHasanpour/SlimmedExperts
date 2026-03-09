import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "slimmed_experts"
PYTHON_VERSION = "3.12"

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run zensical build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run zensical serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

# Code quality commands
@task
def lint(ctx: Context, fix: bool = True) -> None:
    """Run Ruff linter."""
    if fix:
        cmd = "uv run ruff check --fix src/ tests/"
    else:
        cmd = "uv run ruff check src/ tests/"
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def format(ctx: Context, fix: bool = True) -> None:
    """Run Ruff formatter."""
    if fix:
        cmd = "uv run ruff format src/ tests/"
    else:
        cmd = "uv run ruff format --check src/ tests/"
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def typecheck(ctx: Context) -> None:
    """Run MyPy type checker."""
    ctx.run("uv run mypy src/ tests/", echo=True, pty=not WINDOWS)

@task
def quality(ctx: Context) -> None:
    """Run quality checks."""
    lint(ctx, fix=False)
    format(ctx, fix=False)
    typecheck(ctx)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

# Cleaning commands
@task
def clean(ctx: Context) -> None:
    """Clean up build artifacts."""
    ctx.run("rm -rf build .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info", echo=True, pty=not WINDOWS)
    ctx.run("find . -type f -name '*.py[co]' -delete", echo=True, pty=not WINDOWS)
    ctx.run("find . -type d -name '__pycache__' -delete", echo=True, pty=not WINDOWS)
