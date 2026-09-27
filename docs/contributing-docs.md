# Contribute to the docs

The site is built from Markdown in the [package repository](https://github.com/rjn32s/gemmadecision).
Use the pencil icon on a page to propose an edit, or build the site locally.
Building the documentation does not load the model or require a GPU.

## Preview locally

```bash
git clone https://github.com/rjn32s/gemmadecision.git
cd gemmadecision
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

On Windows, activate with `.venv-docs\Scripts\activate` instead. Open the local
address printed by MkDocs. It reloads when Markdown changes.

## Check an edit

```bash
python scripts/check_docs.py
python -m mkdocs build --strict
```

The first command compiles every fenced Python example without executing it.
The strict site build checks navigation and internal Markdown links. For the
shared runnable recipes, install the package test dependencies in a separate
environment and run:

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_documented_examples.py
python examples/recipes.py --list
```

Recipe tests use fake inference and exercise the actual package interfaces.
They verify API usage, not prediction quality. Running a recipe with
`--recipe` performs real inference and can download the model.

## Add a recipe

1. Create a page under `docs/use-cases/` with a concrete task and a complete example.
2. Explain the return value and what the application does with it.
3. Add links from `mkdocs.yml` and the [recipe index](use-cases/index.md).
4. Add a runnable example or mocked check when introducing a new API pattern.

Keep imports in each Python block so readers can copy a block on its own.
Use original, synthetic example inputs. Avoid presenting invented predictions
as measured outputs or recipe ideas as evaluated model capabilities. Link
performance statements to a reproducible measurement with its hardware and input size.

## Publication

The `Documentation` GitHub Actions workflow builds pull requests and publishes
successful builds from `main` to GitHub Pages. A documentation change does not
require a new PyPI release. Package and model versions are separate; update
examples when either interface changes.
