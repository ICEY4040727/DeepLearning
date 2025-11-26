# DeepLearning

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ICEY4040727/DeepLearning/HEAD)
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/ICEY4040727/DeepLearning)

A collection of Jupyter notebooks for deep learning study and experiments.

## Quick Start

### Run in the Cloud (No Installation Required)

- **Binder**: Click the badge above or [launch on Binder](https://mybinder.org/v2/gh/ICEY4040727/DeepLearning/HEAD) to run notebooks in your browser.
- **Gitpod**: Click the Gitpod badge or [open in Gitpod](https://gitpod.io/#https://github.com/ICEY4040727/DeepLearning) for a VS Code-like cloud workspace with JupyterLab.

### Local Development

1. Clone the repository
2. Create a conda environment:
   ```bash
   conda env create -f binder/environment.yml
   conda activate binder-env
   ```
3. Start JupyterLab:
   ```bash
   jupyter lab
   ```

### VS Code Remote Containers / GitHub Codespaces

This repository includes a devcontainer configuration. Open the repository in VS Code with the Remote - Containers extension, or launch it directly in GitHub Codespaces.

## GitHub Pages

The notebooks are automatically converted to HTML and published to GitHub Pages. To enable:

1. Go to **Settings** → **Pages**
2. Set **Source** to `main` branch and `/docs` folder
3. Save

The documentation will be available at: `https://ICEY4040727.github.io/DeepLearning/`

## Pre-commit Hooks

This repository uses `nbstripout` to automatically strip notebook outputs before commits. To enable:

```bash
pip install pre-commit
pre-commit install
```

## Contents

- `预备知识/` - Prerequisite knowledge notebooks covering data operations, linear algebra, etc.
- Additional notebooks and Python scripts for deep learning experiments.
