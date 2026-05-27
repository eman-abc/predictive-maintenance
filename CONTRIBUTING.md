# Contributing (solo workflow)

This project uses a simple trunk-based flow: **`main` stays deployable**, and all work happens on short-lived branches.

## Branch naming

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `chore/` | Tooling, CI, dependencies |
| `test/` | Test-only changes |

Examples: `feat/lstm-hyperparameter-tuning`, `fix/alert-threshold-edge-case`.

## Day-to-day workflow

```bash
git checkout main
git pull origin main
git checkout -b feat/your-topic

# Make changes; commit often with conventional messages:
#   feat: add ...
#   fix: ...
#   docs: ...
#   test: ...
#   chore: ...

git push -u origin feat/your-topic
```

Open a pull request on GitHub targeting `main`, wait for **CI** (pytest) to pass, then merge.

## Merge strategy: squash

Use **Squash and merge** on every PR so `main` gets one clean commit per change (e.g. `feat: add Cox PH survival training`).

- Keeps `main` history readable
- Hides noisy WIP commits from the default log
- Each squash commit should match [Conventional Commits](https://www.conventionalcommits.org/)

After merging, delete the remote branch and update locally:

```bash
git checkout main
git pull origin main
git branch -d feat/your-topic
```

## Commit messages

```
<type>: <imperative summary>
```

| Type | When |
|------|------|
| `feat` | New behavior |
| `fix` | Bug fix |
| `docs` | Docs / notebooks only |
| `test` | Tests only |
| `chore` | Config, deps, CI |
| `refactor` | No behavior change |

## CI

Every push and PR to `main` runs `.github/workflows/ci.yml` (`pytest` on Python 3.12).

Before opening a PR locally:

```bash
pytest
```

## What not to commit

- `.env` (secrets)
- `data/raw/` datasets
- Trained artifacts in `models/` (`*.pkl`, `*.pt`, `*.joblib`)
- `.venv/`, `__pycache__/`, `mlruns/`

See `.gitignore` for the full list.
