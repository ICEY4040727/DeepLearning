说明
----
本仓库用于保存与演示深度学习相关的 Jupyter Notebook 与示例代码。为了方便在云端与本地进行开发、复现与展示，我们为仓库提供了以下支持（适用于公开仓库）：

- 在浏览器中直接运行 JupyterLab（Binder）
- 在浏览器中获得 VS Code-like 工作区（Gitpod）
- 在 Codespaces / Remote-Containers 中重用的 devcontainer
- 在本地 commit 前自动清除 Notebook 输出（pre-commit + nbstripout）
- GitHub Actions：把 `notebooks/` 下的 `.ipynb` 自动转换为 HTML，生成到 `docs/` 并可通过 GitHub Pages 发布

本 README 面向两类读者：
1. 只是想看、运行或阅读 Notebook 的访客（Usage / 运行示例）  
2. 想为项目贡献代码或 Notebook 的开发者（Contributor / 开发要求与标准）

快速开始（访客 / 使用者）
-----------------------
如果你只想在浏览器中查看或运行 Notebook（无需在本地安装复杂环境），请选择 Binder 或 Gitpod：

- Binder（运行 JupyterLab / Notebook，适合交互式演示，完全免费但为短期会话）  
  点击或访问：  
  https://mybinder.org/v2/gh/ICEY4040727/DeepLearning/HEAD

- Gitpod（浏览器中的 VS Code-like 体验，支持扩展、调试、终端等）  
  打开仓库：  
  https://gitpod.io/#https://github.com/ICEY4040727/DeepLearning

如果你想在本地运行（开发者 / 贡献者）
1. 克隆仓库并切换到新分支（请不要直接在 main 分支开发）：
   ```
   git clone https://github.com/ICEY4040727/DeepLearning.git
   cd DeepLearning
   git checkout -b feature/your-change
   ```

2. 建议使用可编辑安装（editable install）方式管理 src/ 下的可复用包  
   - 目录建议：将可复用模块放到 `src/yourpkg/...`，仓库根放 `setup.py`（仓库已包含示例 setup.py）。  
   - 可编辑安装能让你在开发时直接在 Notebook 中导入并即时看到代码的修改，无需反复 reinstall。

3. 在本地创建并激活虚拟环境（推荐）
   - macOS / Linux:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows PowerShell:
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

4. 安装依赖并做可编辑安装
   ```
   pip install --upgrade pip
   pip install -r requirements.txt     # 若存在
   pip install -e .
   
   安装 Torch（必需）
- 默认推荐安装 CPU 版本（适用于大多数环境、Binder/Gitpod/DevContainer/Codespaces）：
  - pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
- 如果你需要 GPU 加速，请根据你的 CUDA/ROCm 版本选择官方命令（请参考 PyTorch 官网安装页面）：
  - https://pytorch.org/get-started/locally/
  - 示例（CUDA 12.x，请以官网为准）：  
    pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.2"
说明：
- Binder/Gitpod 默认使用 CPU 版本；GPU 版仅在本地或支持 GPU 的特定环境可用。
- 如果你直接使用 requirements.txt 安装（pip install -r requirements.txt），它会拉取 PyPI 上的 CPU 版 wheels（通常较快）。
   ```
   

5. 为 Jupyter 注册并使用该虚拟环境的 kernel（推荐）
   ```
   pip install ipykernel
   python -m ipykernel install --user --name=deepenv --display-name "Python (deepenv)"
   ```
   启动 JupyterLab 后，在 Kernel 列表中选择 "Python (deepenv)"，Notebook 将使用你本地虚拟环境并能 import src/ 中的包。

项目约定与目录结构
-------------------
- notebooks/            —— 所有用于自动构建与展示的 Jupyter Notebook（CI 将只处理此目录）
- src/                  —— 可复用的 Python 模块 / 包（供 notebook 导入）
- data/                 —— 小型示例数据（不要放大文件）
- binder/               —— Binder 环境配置（environment.yml 等）
- .devcontainer/        —— DevContainer 配置（用于 Codespaces / Remote-Containers）
- .github/workflows/    —— GitHub Actions workflow（自动把 notebooks/ 转为 docs/）
- docs/                 —— 由 Actions 生成的 HTML 展示文件（不要手动修改此目录的自动生成内容）

为什么使用 `pip install -e .`（可编辑安装）
- 开发时修改 src/ 里的代码会立即反映在 Notebook 中，无需重新安装包。
- 云端环境（Binder / Gitpod / devcontainer）在初始化时也会执行可编辑安装，以确保 Notebook 能 import 本仓库中的包。

数据与大文件的处理
-------------------
- 请不要将大数据集或敏感数据直接提交到仓库。建议把数据放到云端并在 notebook 中提供下载脚本（或一个 `scripts/download_data.sh`）。
- 在仓库根加入 `.gitignore`，排除大文件夹。

预提交钩子（保持 Notebook 干净）
---------------------------
我们使用 `pre-commit` + `nbstripout` 来在 commit 前自动去除 Notebook 输出，保证仓库的 Notebook 只包含源代码和必要的元信息。

- 本地安装与启用：
  ```
  pip install pre-commit
  pre-commit install
  ```

CI 与自动生成文档（维护者说明）
-------------------------------
- 当 `main` 分支发生 push 时，GitHub Actions 会把 `notebooks/` 下的 `.ipynb` 转为 HTML，并把结果写入 `docs/`（保留目录结构），然后将 `docs/` 提交回仓库，供 GitHub Pages 发布。

启用 GitHub Pages（一次性操作）
- Settings -> Pages -> Source 选择 Branch: `main` 和 Folder: `/docs`，保存。
- 等 Actions 成功运行并将 HTML 写入 `docs/` 后，页面会在 https://ICEY4040727.github.io/DeepLearning/ 可访问。

贡献指南（简要）
-----------------------
1. Fork 或直接在分支上开发（创建 feature 分支）
2. 将 Notebook 放入 `notebooks/`，可复用代码放入 `src/`
3. 本地使用虚拟环境并执行 `pip install -e .`，注册并使用对应 kernel
4. 安装并启用 pre-commit，提交并发起 PR

其他说明
--------
- 若需要我为你把这些文件提交到 `cloud/dev-setup` 并打开 PR，请把我添加为协作者或授权写入；我也可以只 review 你发起的 PR。
- 如果你更愿意使用 pyproject.toml（poetry/PEP 621）而非 setup.py，我可以为你生成对应配置。
