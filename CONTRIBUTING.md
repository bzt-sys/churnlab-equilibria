# Contributing

Thanks for helping improve ChurnLab : Equilibria!

## Setup
1. Create a virtualenv and install in editable mode:
   ```bash
   pip install -e .[dev]
   ```
2. Pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

## Development Workflow
- Branch from `main`, open a PR early, push small commits.
- Add tests for new behavior (or for discovered bugs).
- Keep vectorized paths (CuPy) and CPU fallbacks aligned.
- Prefer pure functions and explicit RNGs (no hidden globals).

## Code Style
- `ruff` for linting/format, `pytest` for tests.
- Docstrings: Google-style, with input/output shapes and dtypes.

## PR Checklist
- [ ] Unit tests pass (`pytest -q`).
- [ ] GPU code path exercised (when applicable).
- [ ] Public API documented in docstrings and README.
- [ ] Performance footnote for O(N) vs O(KN) changes.
- [ ] No breaking changes without deprecation notes.

## Releasing
1. Bump version in `pyproject.toml`.
2. Tag and push: `git tag vX.Y.Z && git push --tags`.
3. Build & upload:
   ```bash
   python -m build
   twine upload dist/*
   ```
