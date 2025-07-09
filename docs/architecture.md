## Executive Summary

Foundry-LLM’s governance model keeps **main** production-ready at all times, funnels all work through short-lived feature branches, and enforces code quality automatically. A single GitHub Actions pipeline runs tests, style checks, and security scans on every pull request; releases are tagged with Semantic Versioning. Code style is handled by **Black** (formatting), **isort** (import ordering), and **Ruff** (ultrafast linting), wired together by **pre-commit** hooks so you never have to think about them again.([black.readthedocs.io][1], [pycqa.github.io][2], [github.com][3], [github.com][4])

---

## 0.1 Repository Bootstrapping

| Step                                                                                | Command / File                                                                                | Purpose                                                                           |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Initialise Git & remote                                                             | `git init && gh repo create Foundry-LLM --private --source .`                                 | Starts version control in place.                                                  |
| Add Python‐specific **.gitignore**                                                  | `curl -o .gitignore https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore` | Keeps caches, venvs, IDE folders out of the repo.([raw.githubusercontent.com][5]) |
| Create a **README.md**, **LICENSE** (MIT recommended), and **CODE\_OF\_CONDUCT.md** |                                                                                               | Documents what the project is and how to contribute.                              |
| Optional: enable **Dependabot** in Settings → “Code security & analysis”            |                                                                                               | Auto-PRs for vulnerable packages.([docs.github.com][6])                           |

---

## 0.2 Branching & Release Flow

### Primary branches

* **main** – always deployable; every commit here is a release candidate.
* **dev** – integration branch; merges from multiple features; squashed into **main** on release.

### Working branches

* **feat/\*** for new functionality
* **fix/\*** for bug patches
* **chore/\*** for build or CI tweaks

The **Feature Branch Workflow** isolates work and keeps *main* green, which is critical for continuous integration.([atlassian.com][7])

#### Merging rules

1. Open a PR from `feat/*` → `dev`.
2. GitHub Actions must pass.
3. At least one approving review (CODEOWNERS can enforce).
4. Squash-merge to keep a clean history.

> **Tip:** If a hotfix is urgent, branch from **main**, merge back to **main**, and cherry-pick into **dev**.

---

## 0.3 Commit Conventions

* Follow the **Conventional Commits** spec:

  ```
  feat(tokenizer): add BPE training pipeline
  fix(trainer): clip gradients to prevent NaNs
  chore(ci): bump ruff to 0.4.0
  ```

  This small discipline lets tooling auto-generate changelogs and determine SemVer bumps.([conventionalcommits.org][8])
* Keep messages ≤ 72 chars in the summary line; wrap the body at 100 chars.

---

## 0.4 Code Style & Static Analysis

### What are Black, isort, Ruff?

| Tool      | Role                                                    | Key Traits                                                        | Docs                                                         |
| --------- | ------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| **Black** | Opinionated formatter: rewrites code to a single style. | Deterministic, “uncompromising.”                                  | ([black.readthedocs.io][1], [github.com][9])                 |
| **isort** | Sorts and groups `import` statements.                   | Alphabetises imports, respects Black’s line-length profile.       | ([pycqa.github.io][2], [here-be-pythons.readthedocs.io][10]) |
| **Ruff**  | Linter & optional formatter written in Rust.            | 10–100× faster than flake8 + plugins; can also run `ruff format`. | ([github.com][3], [astral.sh][11])                           |

All three integrate seamlessly via **pre-commit**; staged files are auto-fixed before every commit.([github.com][4])

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks: [{id: black}]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks: [{id: isort}]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]        # auto-apply safe fixes
```

Activate with:

```bash
pip install pre-commit black isort ruff
pre-commit install
```

---

## 0.5 Continuous Integration (GitHub Actions)

Create `.github/workflows/ci.yml`:

```yaml
name: CI
on: [pull_request, push]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: {python-version: ['3.10', '3.11']}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with: {python-version: ${{ matrix.python-version }}}
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: black --check .
      - run: isort --check .
      - run: pytest -q --cov=foundry_llm
```

* **Ruff-action** can annotate PRs inline.([docs.astral.sh][12])
* Cache `~/.cache/pip` to speed up installs if builds grow.

---

## 0.6 Versioning & Releases

* Tag new releases `vMAJOR.MINOR.PATCH` per **Semantic Versioning 2.0.0**. Breaking public APIs → MAJOR bump; new features → MINOR; fixes → PATCH.([semver.org][13])
* GitHub’s **Release** page + `gh release create` attaches the changelog generated from Conventional Commits.

---

## 0.7 Security & Dependency Hygiene

* Enable **Dependabot security updates**; it will open PRs when PyPI packages have CVE patches.([docs.github.com][14])
* Optionally add a workflow that auto-merges Dependabot PRs after CI passes.([docs.github.com][15])

---

## 0.8 Governance Extras

* **PR template**: checklist for tests, docs, and changelog.
* **CODEOWNERS**: enforce reviews from domain experts.
* **Branch protection rules**: require CI success and one approval before merging to **main**.
