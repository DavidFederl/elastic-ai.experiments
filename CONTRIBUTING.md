# Contributing Guide

Thank you for your interest in contributing to the experiment-runner project!
We welcome contributions from everyone.

## How to Contribute

### Reporting Issues

Before opening a new issue, please:

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** to ensure it's not already covered
3. **Provide detailed information**:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Screenshots if applicable

### Suggesting Features

Feature requests should:

1. **Describe the use case** clearly
2. **Explain the benefit** to the project
3. **Include examples** if possible
4. **Be open for discussion** - we may suggest alternatives

## Development Workflow

### 1. Fork the Repository

```bash
# Fork the repository on GitHub
# Clone your fork locally
git clone https://github.com/your-username/experiment-runner.git
cd experiment-runner
```

### 2. Set Up the Development Environment

Follow the instructions in [SETUP.md](SETUP.md) to set up your development environment.

### 3. Create a Feature Branch

```bash
# Create a new branch for your feature/fix
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions**:

- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation improvements
- `refactor/*` - Code refactoring
- `test/*` - Test improvements

### 4. Make Your Changes

- **Follow existing code style** (see [Code Style](#code-style))
- **Write tests** for new functionality
- **Update documentation** if needed
- **Keep changes focused** - one feature/fix per branch

### 5. Commit Your Changes

```bash
# Stage your changes
git add .

# Commit with a conventional commit message
git commit -m "feat: add new feature description"
```

**Commit message guidelines (Conventional Commits)**:

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Common commit types**:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect meaning (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools and libraries
- `revert`: Reverts a previous commit

**Examples**:

```bash
# Feature with scope
git commit -m "feat(api): add new endpoint for user profiles"

# Bug fix with issue reference
git commit -m "fix: resolve memory leak in data loader"

# Documentation update
git commit -m "docs: add setup guide for Windows users"

# Breaking change with footer
git commit -m "feat!: drop support for Python 3.9

BREAKING CHANGE: Minimum Python version is now 3.10"

# Multiple paragraphs and footer
git commit -m "refactor: optimize delta compression algorithm

- Reduce memory usage by 30%
- Improve processing speed by 25%

Closes #123, Related to #456"
```

**Best practices**:

- Use present tense ("add" not "added")
- Keep first line under 72 characters
- Use lowercase for type and scope
- No period at the end of subject line
- Reference issues in footer: `Closes #123`, `Fixes #456`, `Related to #789`

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

1. Go to the [original repository](https://github.com/es-ude/experiment-runner)
2. Click "New Pull Request"
3. Select your branch from your fork
4. Fill out the PR template completely
5. Wait for review and address feedback

## Pull Request Guidelines

### Before Submitting

✅ **Checklist**:

- [ ] Code follows the project's style guidelines
- [ ] Tests are added/updated and passing
- [ ] Documentation is updated (if applicable)
- [ ] All existing tests pass
- [ ] No breaking changes (or documented if necessary)

### During Review

- **Be responsive** to reviewer feedback
- **Make requested changes** or explain why they're not needed
- **Keep the conversation** constructive and professional
- **Small, focused PRs** are easier to review and merge

### After Approval

- A maintainer will merge your PR
- Your changes will be included in the next release
- **Celebrate!** 🎉 You've contributed to open source!

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Type hints are encouraged
- Docstrings for public functions and classes

```bash
# Check and fix code style
uv run ruff check --fix
uv run ruff format
```

### Documentation

- Use clear, concise language
- Keep docstrings up to date
- Use Markdown for documentation files
- Include examples where helpful

### Testing

- Write tests for new functionality
- Follow existing test patterns
- Aim for good coverage
- Use descriptive test names

```bash
# Run tests
uv run pytest
```

## Development Standards

### Branch Protection

- **`main` branch is protected** - no direct pushes allowed
- All changes must go through pull requests
- At least one approval required for merging
- All checks must pass before merging

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR` - Breaking changes
- `MINOR` - New features (backward compatible)
- `PATCH` - Bug fixes (backward compatible)

## Getting Help

### Need Assistance?

- **Check existing issues** - Your question may already be answered
- **Ask in discussions** - GitHub Discussions for general questions
- **Open an issue** - For specific problems or bugs
- **Join our community** - Check README for community links

## Recognition

All contributors will be:

- Added to the contributors list
- Recognized in release notes
- Appreciated by the community 🙏

Thank you for contributing to experiment-runner!
Your help makes this project better for everyone.

---

_Inspired by other great open source projects and their contribution guidelines._
