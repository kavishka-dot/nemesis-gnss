# Contributing to NEMESIS-GNSS

Thank you for your interest in contributing.

## Development Setup

```bash
git clone https://github.com/kavishka-dot/nemesis-gnss.git
cd nemesis-gnss
pip install -e ".[all,dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

This project uses `black` for formatting and `ruff` for linting:

```bash
black nemesis/
ruff check nemesis/
```

## Submitting a Pull Request

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes with a clear message
4. Ensure all tests pass
5. Open a pull request against `main`

## Reporting Issues

Please use the [GitHub Issue Tracker](https://github.com/kavishka-dot/nemesis-gnss/issues).
Include your OS, Python version, and `nemesis info` output.

## License

By contributing, you agree your contributions will be licensed under Apache 2.0.
