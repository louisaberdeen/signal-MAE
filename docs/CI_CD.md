# CI/CD Pipeline

This project uses GitHub Actions for continuous integration.

## Overview

Tests run automatically on:
- Push to `main` branch
- Push to `dev` branch
- Pull requests targeting `main` or `dev`

## Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Push to   │────▶│  Run Tests  │────▶│   Pass/Fail │
│  main/dev   │     │             │     │   Status    │
└─────────────┘     └─────────────┘     └─────────────┘
```

## What Gets Tested

1. **Test Generation** - Runs `python tests/generate_tests.py`
2. **Plugin Tests** - Runs `pytest tests/generated/ -v`
   - Model interface compliance
   - Data loader interface compliance
   - Transform interface compliance
   - Architecture compatibility (various input sizes)
3. **Registry Verification** - Confirms all plugins are registered

## Workflow File

Located at `.github/workflows/tests.yml`

### Triggers

```yaml
on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
```

### Steps

| Step | Description |
|------|-------------|
| Checkout | Clone repository |
| Setup Python | Install Python 3.11 |
| Cache | Cache pip dependencies |
| Install | Install torch, pytest, numpy, einops, tqdm |
| Generate Tests | Run test generator for plugins |
| Run Tests | Execute pytest on generated tests |
| Verify Plugins | Confirm registries have expected plugins |

## Local Testing

Run the same tests locally before pushing:

```bash
# Activate environment
source .venv/bin/activate

# Generate tests
python tests/generate_tests.py

# Run tests
python -m pytest tests/generated/ -v

# Verify plugins
python -c "from src import model_registry; print(model_registry.list())"
```

## Branch Strategy

```
main (production)
  ↑
  │ PR with passing tests
  │
dev (development)
  ↑
  │ PR with passing tests
  │
feature/cleanup/* (feature branches)
```

## Adding New Tests

When adding new plugins to `src/`, the test generator automatically creates tests:

1. Add your plugin to `src/models/`, `src/data/`, or `src/transforms/`
2. Register it with the appropriate decorator
3. Import it in the module's `__init__.py`
4. Run `python tests/generate_tests.py` to create tests
5. Push to `dev` - CI will run the new tests

## Troubleshooting

**Tests fail on CI but pass locally:**
- Ensure all imports are correct (no local-only paths)
- Check Python version compatibility (CI uses 3.11)
- Verify all dependencies are installed in CI

**Plugin not registered:**
- Ensure class is imported in `__init__.py`
- Check decorator syntax: `@registry.register("name")`
