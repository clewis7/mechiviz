# branchpoint
Interactive mechanistic interpretability 


## Installation 

1. Clone the repo 
```bash
git clone https://github.com/clewis7/branchpoint.git
cd branchpoint
```
2. Install deps using [uv](https://docs.astral.sh/uv/)
```bash
uv sync --extra dev 
```

## For developers

```bash
uv sync --extra dev    # to get dev dependencies
uv lock --upgrade      # to update deps when necessary

# Common tasks:
uv run ruff format     # auto-format
uv run ruff check      # lint
uv run pytest tests    # run unit tests

```