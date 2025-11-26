# DeepLearning

A repository for deep learning experiments and Jupyter notebooks.

## Quick Start

### Launch in Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICEY4040727/DeepLearning/main)

Click the badge above to launch a Jupyter Lab environment in your browser.

### Launch in Gitpod
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/ICEY4040727/DeepLearning)

Click the badge above to open the repository in a VS Code-like cloud workspace.

## Development

### GitHub Codespaces / Dev Containers

This repository includes a `.devcontainer` configuration for use with:
- [GitHub Codespaces](https://github.com/features/codespaces)
- [VS Code Remote - Containers](https://code.visualstudio.com/docs/remote/containers)

Simply open the repository in Codespaces or use the "Reopen in Container" command in VS Code.

### Pre-commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) with [nbstripout](https://github.com/kynan/nbstripout) to automatically strip notebook outputs before commits.

To set up pre-commit hooks locally:

```bash
pip install pre-commit
pre-commit install
```

## GitHub Pages

This repository uses GitHub Actions to automatically convert Jupyter notebooks to HTML and publish them to the `docs/` folder on each push to the `main` branch.

To enable GitHub Pages:
1. Go to repository **Settings** > **Pages**
2. Under **Source**, select **Deploy from a branch**
3. Select the `main` branch and `/docs` folder
4. Click **Save**

Your notebooks will be available at: `https://ICEY4040727.github.io/DeepLearning/`

## Installation

```bash
pip install -r requirements.txt
```
