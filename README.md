# intervene

偏离点涂黑干预工具。该目录有独立 Git 历史，可单独克隆，也可作为 `vln/intervene` submodule 使用。协作者只需本仓库的访问权限。

## 单独使用

本地仓库源目前为 `/home/jwm/workspace/git/intervene.git`：

```bash
git clone /home/jwm/workspace/git/intervene.git intervene
cd intervene
python3 -m pip install -e . --no-deps --no-build-isolation
python3 -m intervene.occlusion --help
python3 -m pytest -q tests
```

安装命令沿用当前环境的依赖，不下载或替换各模型的包。核心代码需 NumPy；测试需 pytest；图片功能需 Pillow。完整运行须在对应模型环境分别安装本包，并准备 Habitat、模型工程、权重和数据。需要由 pip 安装核心依赖时，可使用 `python -m pip install -e .`。

无需安装的开发入口是仓库根目录下的 `python -m occlusion --help`。安装后的 `python -m intervene.occlusion` 和 `intervene-occlusion` 可从其他目录使用。

## 外部输入配置

独立克隆不要求访问整个 `vln` 仓库。将所需数据放在可访问的目录，并在启动 Python 前设置以下环境变量。路径可为绝对路径，也支持 `~`；相对路径以启动目录为基准。

| 变量 | 用途 | 未配置时的兼容默认值 |
| --- | --- | --- |
| `INTERVENE_MODELS_ROOT` | 各模型原工程的父目录 | 原 workspace 同级的 `opensource/` |
| `INTERVENE_DATASET_ROOT` | 含 `<split>/<split>.json.gz` 的 R2R 数据集根 | 原 workspace 同级的 `dataset/R2R_VLNCE_v1-3_preprocessed/` |
| `INTERVENE_EXPORT_ROOT` | 人工标注与失败列表的导出根 | 原 workspace 下的 `results/export/` |
| `INTERVENE_ARCHIVE_ROOT` | 已准备好的正式原轨迹与实验归档 | `/media/jwm/大数据/result` |
| `INTERVENE_RESULTS_ROOT` | 本地统计和 HTML 报告输出 | 本仓库 `results/` |
| `INTERVENE_WORKSPACE_ROOT` | 原 workspace；仅为兼容旧目录布局及默认预测路径 | 本仓库父目录 |
| `INTERVENE_ENV_FILE` | 语义区域 API 的凭据文件 | 优先本仓库 `.env`，否则 workspace 的 `.env` |

可复制 `.env.example` 为 `.env`，填写后加载：

```bash
set -a
. ./.env
set +a
python -m intervene.occlusion --help
```

路径在模块导入时读取。语义模块自动读取 `.env` 的功能只用于补充 API 凭据；路径变量须在进程启动前加载。程序不会因配置归档目录而自动创建空归档代替已有数据。

模型工程不在公共根下时，在批次 JSON 的 `adapter.root` 指定实际位置。OmniNav 的原始预测位置通过 `selection.predictions` 指定；原始统计通过 `selection.stats_dir` 指定。其他标注或日志位置通过 `selection.reviews`、`selection.failures`、`selection.logs` 覆盖。SimpleVLN 的选点校验还从 `INTERVENE_MODELS_ROOT/SimpleVLN/eval/vln/result_store.py` 读取辅助代码，因此也需设置正确的模型根目录。所有批次输入必须属于同一 split。

默认标注仍指向 `val_unseen`；ETPNav 原日志文件名仍为 `val_unseen`；OmniNav 原生任务配置可能直接写死 dataset/GT 路径。运行 train 时需准备对应材料并调整这些配置/读取逻辑，独立仓库拆分不会自动补齐 train 数据。

详细数据需求、JSON 配置、`plan → offline/run → analyze → viz` 命令和 ETPNav 的 `none` 前置分支见 [occlusion/README.md](occlusion/README.md)。

## 协作与版本

日常代码修改、提交和分支操作均在本仓库内进行：

```bash
git switch -c my-change
python3 -m pytest -q tests
git add <changed-files>
git commit -m "Describe the change"
git push -u origin my-change
```

父仓库维护者在 `vln` 中用 `git add intervene` 记录选定的子仓库提交。依赖数据、权重、`351/` 本地材料、虚拟环境、API 凭据、`provider.json` 和 `*.local.json` 配置均不纳入新提交；已保留的历史内容仍属于本仓库历史，发布前可单独审阅。

目前 origin 仅在本机，无网络发布或托管权限配置。将来建立私有协作远端后，在本仓库修改 origin；同时由父仓库维护者修改 `.gitmodules` 的 `submodule.intervene.url` 并运行 `git submodule sync -- intervene`。只向协作者授予独立仓库的权限。

本地 submodule 初始化需要按命令允许 file 协议，无需改变全局 Git 设置：

```bash
# 在父仓库中执行
git -c protocol.file.allow=always submodule update --init -- intervene
```
