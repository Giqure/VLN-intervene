# 偏离点遮挡干预（C6002）

实现依据：仓库根目录的 `proposal.md`。原始评测、人工标注、模型输出保持原样；仅按方案向正式归档的原轨迹追加/更新 `deviation` 记录。

## 模块

- `core/`：NumPy 几何、相机、区域、tracker、逐步干预、统计、归档文件契约。兼容 Python 3.6 / NumPy 1.19。
- `select/`：按模型从原始轨迹、标注和原轨迹输出选点，不重新前向模型。
- `agents/`：三个模型的原生配置与闭环适配器，分别在其自身环境中运行。
- `runner.py`、`cli.py`：`plan → offline / run → analyze → viz` 文件交接。
- `offline.py`：按原轨迹观测位姿渲染，可逐像素检查附加 tracker 深度相机前后的原 RGB。
- `viz/`：批次报告与逐条人工复核页面。
- `configs/`：JSON 配置模板及区域/tracker 配置片段；不是可直接启动的正式实验参数。
- 仓库根目录 `tests/`：路径隔离、报告输出和纯合成涂黑测试，不代表实际模型实验结果。

## 环境与先决条件

先在当前独立仓库根目录执行 `python -m pip install -e . --no-deps --no-build-isolation`，之后可从任意目录运行 `python -m intervene.occlusion`。不安装时，可在本仓库根目录运行 `python -m occlusion`。外部路径配置见根目录 [README](../README.md)。

| 模型 | 本地原工程 | 环境 |
| --- | --- | --- |
| OmniNav | `/home/jwm/workspace/opensource/OmniNav/infer_r2r_rxr` | `omninav` |
| SimpleVLN | `/home/jwm/workspace/opensource/SimpleVLN/eval` | `simplevln-eval` |
| ETPNav | `/home/jwm/workspace/opensource/ETPNav` | `etpnav` |

模型依赖、权重、数据集、Habitat 及驱动沿用各自原工程，不在 core 中混装。core 仅依赖 NumPy；完整测试还需 pytest，图片保存需 Pillow。适配器使用模型原生评测和度量。

正式归档根通过 `INTERVENE_ARCHIVE_ROOT` 配置；未配置时沿用 **`/media/jwm/大数据/result`**。目录必须已存在，且存在 `checksums.jsonl`、`manifest.json`、`README.md`、`settings.json`。程序不会在别处回退生成正式归档。`adapter.scratch` 必须是独立的绝对路径，不可指向原始评测或原始模型输出目录。

SimpleVLN 还需要可访问的 vLLM OpenAI 兼容服务，以及与当前 split 匹配的原生 R2R GT 文件。`gt_path` 支持 `{split}` 占位符，不能用其他数据集的 GT 代替。原记录缺少 nDTW 时，用原生 NDTW measure 对归档 positions 计算，不补零。

## 配置

先复制 `configs/<model>-<arm>.template.json` 到自己的配置文件，再填完所有 `REQUIRED_*`、`/ABSOLUTE/...` 占位符。模板使用合法 JSON，但故意保留必填参数，不能作为正式实验默认值。

- `model`：`omninav` / `simplevln` / `etpnav`；`arm`：`none` / `rgb` / `rgbd`，其中 `rgbd` 仅适用于 ETPNav。
- `channels` 必须与 arm 对应：`[]` / `["rgb"]` / `["rgb", "depth"]`。
- `offset`：整数，涂黑步为偏离点换算后的步号加 offset；该步缺少所需原输出时排除。
- `follow=0`：只涂黑 start；`follow=N`：包含 start 到 `start+N`；`follow=null`：跟到 episode 结束。
- `depth_bias`：必填、非负、单位米；只用于 tracker 可见性判断。
- `depth_max`：必填、正数、单位米；追踪深度相机的量程，与模型自己读的深度量程无关。超过量程的射线记为没打到，只保留方向。
- `adapter.history`（仅 OmniNav，必填）：`clean` 表示模型历史里保留真实观测，只有当步输入被涂黑（SimpleVLN 固定如此）；`blackened` 表示历史里存的是涂黑后的图。两种结果不同，所以写进批次参数。
- `save_images=0` 不保存闭环涂黑图；正整数限制保存的涂黑步数；负数保存全部涂黑步。
- `adapter`：模型原工程根、配置、独立 scratch，以及各自权重/服务/GT 参数。相机参数从模型原生配置读取并固化进批次，不手填另一套相机常量。

方位带配 `heading` 或 `surface` tracker，有四种 `region_spec` 写法。角度单位为度，正值为左（逆时针）；"th4 坐标系"指相对涂黑步观测时机器人朝向的角度。

| `region_spec` | 涂黑范围 |
| --- | --- |
| `{"kind": "heading", "half": 15.0}` | th4 ± half，对称（旧写法，仍可用） |
| `{"kind": "heading", "bounds": [c, l, r]}` | `[c−r, c+l]`；`c` 为 `null` 或 `"th4"` 时取候选的 th4，数值时为 th4 坐标系下的角度；`l`、`r` 非负 |
| `{"kind": "heading", "bounds": [l, r]}` | 带符号角度，区间 `[r, l]`（要求 `r ≤ l`），不以 th4 为中心 |
| `{"kind": "columns", "view": "<view名>", "bounds": [l, r]}` | 该 view 掩码图的第 `l`～`r` 列（从 0 起，左边缘为 0，闭区间，顺序不限） |

`columns` 在第一个涂黑帧把整列像素换算成世界朝向区间，之后与 `heading` 区域相同，随机器人位姿跟随变化。`view` 必须显式给出，列号须小于该 view 的 `mask_hw[1]`；`plan` 阶段会按 rig 检查。选点阶段（`select/`）的可见性筛选不随 `region_spec` 改变。错误过道 `kind=corridor` 只配 `volume` tracker。替换配置片段时，要整体替换 `region_spec` 和 `tracker_spec`，不是追加冲突字段。

### 语义模型给出涂黑范围（`kind=model`）

`{"kind": "model", "options": {...}}`：在第一个涂黑帧调用 API，模型返回的列区间按 `columns` 处理，之后同样跟随位姿。配 `heading` 或 `surface` tracker，片段见 `configs/model.fragment.json`。

- 传输：Responses API，`POST {base_url}/v1/responses`，请求带自定义 `User-Agent`（网关会拦 urllib 的默认值，HTTP 403 / 1010）。响应的 `model` 或 `reasoning.effort` 与请求不符时直接报错，不当作成功。
- 密钥：`api_key_env` 只写环境变量名，值放仓库根目录 `.env`（已在 `.gitignore`），shell 中已设置的同名变量优先。批次参数和记录里不出现密钥。
- `request=views`：发送内容只有 `prompt`、各 `views` 的图像，以及每张图前的一行 `view=<名> hfov=<度> center=<度>`。`prompt` 必填，没有隐含默认值，且必须包含 `{direction}` 和 `{instruction}`：前者替换为原模型这一步的计划方向（相对机器人当前朝向的角度，度，正为左，保留一位小数），后者替换为该 episode 的指令原文（`plan` 阶段从预处理 R2R-VLNCE 表取出，写入候选的 `extra.instruction`；换用 `plan` 之前生成的批次需重新 `plan`）。`hfov` 是该 view 的水平视场角，`center` 是画面正中心对应的方向，约定同上。除方向和指令外，不发送标注类别、偏离步号、GT 路径等任何候选字段。图宽必须等于该 view 的 `mask_hw[1]`。
- `parser=x_range`：要求输出里恰有一个 `x_range=[左, 右]`（0~1 归一化横坐标），可带 `view=<名>`；只发了一个 view 时可省略。换算为 `[floor(左×W), ceil(右×W)−1]` 列。格式不符、越界或多个范围一律报错，该候选记为失败，不退回默认区域。`prompt` 需要要求模型按这个格式回答。
- `reasoning_effort` 必填，不设默认值（该网关不指定时为 medium）。
- 记录：第一个涂黑步的 `blackout` 记录里有 `region` 字段，包含提示词、图像 sha256、模型原始回答、实际模型名与 effort、token 用量和解析出的 spec。
- 换请求方式或解析方式（例如以后对比 mask 输出）：在 `core/semantic.py` 的 `REQUESTS` / `PARSERS` 里注册新函数，再在 `options` 里改名字。

| 参数 | 含义与单位 |
| --- | --- |
| `rho`、`hmin`、`hmax`、`r_near`、`length`、`still` | 过道半径、高度范围、近处排除距离、折线长度、原地阈值，均为米 |
| `gap_max` | Surface 相邻锚点的最大朝向间隔，**度** |
| `near_z` | Surface 相机系深度下限，米 |
| `ds` | Volume 沿射线采样步长，米 |
| `unseen_step` | Heading 未见方向射线间隔，度；显式 `null` 表示不补 |
| `rows` | Heading 可为 `axis` 或正整数；Surface 必须为正整数，表示深度行采样间隔 |

上述几何数值无隐含实验默认值，必须按批次明确给出。Heading 还必须显式设置 `occlusion`、`incremental` 和 `unseen_step`。`surface.fragment.json`、`corridor.fragment.json` 提供所需字段，不提供实验数值。

## 运行

以下 `<...>` 是待替换的文件/批次标识，不是字面命令参数。`plan` 返回的 `batch` 是后续阶段使用的不可变批次名（包括参数 hash）。模型相关阶段使用相应 conda 环境。

```bash
# 例如 OmniNav；SimpleVLN 将环境与配置换成对应模型
conda run -n omninav python -m intervene.occlusion plan --config <filled-config.json>

# 不前向模型：按原轨迹渲染 clean / black / mask，并核对原 RGB
conda run -n omninav python -m intervene.occlusion offline \
  --model omninav --batch <batch> --check-rgb

# 真正闭环运行
conda run -n omninav python -m intervene.occlusion run \
  --model omninav --batch <batch>

# 统计与页面（只需 NumPy，不加载模型）
python3 -m intervene.occlusion analyze --model omninav --batch <batch>
python3 -m intervene.occlusion viz --model omninav --batch <batch>
```

`offline` 和 `run` 都支持重复 `--episode <id>` 以及 `--limit <positive-int>`。`run --fail-fast` 在首个候选错误后停止；未指定时记录错误后继续其他候选。`run --retry-errors` 用于 SimpleVLN 重跑出错的候选。`analyze` 的统计选项为 `--n-boot`、`--level`、`--seed`，CLI 默认值分别为 1000、0.95、0；按 trajectory_id 成簇 bootstrap，不把同轨迹的多个指令当独立轨迹。

### ETPNav 顺序

必须先运行无干预分支，记录 planning provenance 并核对完整复现；只有 `full_same` 的记录可作为后续干预的 ghost 来源。

```bash
conda run -n etpnav python -m intervene.occlusion plan --config <filled-etpnav-none.json>
conda run -n etpnav python -m intervene.occlusion run --model etpnav --batch <none-batch>

conda run -n etpnav python -m intervene.occlusion plan \
  --config <filled-etpnav-rgb.json> --none-batch <none-batch>
conda run -n etpnav python -m intervene.occlusion run --model etpnav --batch <rgb-batch>

conda run -n etpnav python -m intervene.occlusion plan \
  --config <filled-etpnav-rgbd.json> --none-batch <none-batch>
conda run -n etpnav python -m intervene.occlusion run --model etpnav --batch <rgbd-batch>
```

ETPNav 的注入点在每次规划的原生 `batch_obs` 之前；RGBD 对 waypoint predictor 和 planner 使用同一份已涂黑深度。tracker 读取独立、未修改的米制深度，不把模型已置零的深度再反馈给可见性判断。

## 步号、重跑与失败

OmniNav / SimpleVLN 归档第 k 步记录的位姿在动作之后，所以第 k 次观测使用 ready（k=1）或第 k−1 步位姿。ETPNav decision 直接是规划时位姿，不错位。SimpleVLN 在 start 前回放原动作、不调用服务采样；start 清空 pending 队列，历史图像始终保存 clean RGB。

OmniNav 与 ETPNav 是确定性的：出错或中断留下的 attempt 在下次 `run` 时直接重跑。SimpleVLN 是采样的，每候选只允许一次 attempt，出错或中断后不会自动重采样；确认要重跑时加 `run --retry-errors`。已成功保存且身份一致的 `run.json` 可被续跑跳过。错误进入归档错误记录；分析显式报告缺失、错误、attempt 与截断，不用零值补结果。`none` 不执行涂黑，但仍核对 start 到达与全程复现。

同参数 `plan` 可重复运行，既有批次参数与候选不可变。原轨迹的偏离记录通过追加/原子补字段维护，原有事件字节不重写。模型前缀、复现信息、终止原因及干预日志写入分支记录；max_len、forced/early stop、无可选点等不会伪装成模型自然 STOP。

## 文件位置与归档维护

- 正式批次：`<archive>/experiments/<model>/intervene/<series>/<batch>/`，保存参数、candidates、plan、offline 等。series 分别为 `C6002o`、`C6002s`、`C6002e`。
- 分支 run、attempt、图片位于归档模型 episode 目录的干预子目录；沿用归档索引中的 `episodes[].directory`，不另造 episode 目录。
- 本仓库 `results/<model>/<batch>/`（可用 `INTERVENE_RESULTS_ROOT` 覆盖） 只保存 `summary.json`、`index.html`、`review.html`。页面链接归档中的原始图片，不把 run 或图片复制到本地 results。
- 批次有独占锁，JSON/JSONL 原子写入，归档变更先记入各系列 `records.jsonl`。

批次阶段结束、无其他进程写正式归档时，用一次维护命令汇总所涉及模型：

```bash
python3 -m intervene.occlusion commit-meta \
  --model omninav --model simplevln --model etpnav
```

维护先在 `maintenance/c6002-YYYYMMDD/` 保存根 checksums / manifest / README / settings 的 before 备份，再更新元数据。当天维护目录已存在会拒绝，不覆盖备份。`commit-meta` 是独立维护阶段，不能与 plan/run/offline 并发。

## 验证与当前运行边界

```bash
python3 -m compileall -q occlusion
python3 -m pytest -q tests
python3 -m intervene.occlusion --help
```

实现阶段已验证三套原生配置可加载、三环境 core 合成自测可运行；这些检查不代表 Habitat 真渲染、RGB 逐像素一致或模型闭环已通过。当前工作环境正式归档盘未挂载，`nvidia-smi` 无法连接驱动，且正式几何参数、模型服务与部分外部输入仍需按实际批次填写。因此尚无真实闭环实验结果，也未生成替代正式归档。恢复归档盘与驱动、准备齐真实配置后，按上述顺序执行 `plan`、带 `--check-rgb` 的 `offline`、`run`，再分析与复核。
