# 偏离点遮挡干预：工作方案

## 一、目标

对三个 VLN 智能体（OmniNav、SimpleVLN、ETPNav）的失败 episode 做同一类干预：在人工标注的偏离点，把观测里的一块区域涂黑，并在之后若干步内跟随这块区域持续涂黑，然后闭环跑到结束，和原轨迹比较 SR、SPL、nDTW。目的是检验遮住"走错的那个方向"能否把智能体拉回正确路线。

涉及的变量有以下几项，可以组合，但跟随方式要和区域对应：按朝向和按表面跟随只用于方位带，按体积跟随只用于错误过道。

- **模型**：OmniNav、SimpleVLN、ETPNav。
- **区域**：方位带（中心 ±15°）、错误过道（三维区域）。
- **跟随方式**：按朝向跟随（判遮挡或不判），按表面或体积跟随。
- **跟随步数**：0、若干步、跟到 episode 结束。
- **涂黑步**：偏离点 d，或偏移 offset 步后的 d + offset。

OmniNav 的选点要用原轨迹上的模型输出（d 处的预测方向 th4），读原指令下的前向输出（`interpret/data/swap_dev`、`swap_all` 的 `outputs.npy`，`true` 条件），它覆盖全部 555 条偏离点。涂黑步取 d + offset 时，这一步没有输出的不纳入。

## 二、流程

```
输入（原有内容不改）            plan            run               analyze          viz
原始评测轨迹 ─┐
人工偏离点标注 ┼→ candidates.jsonl → run.json（每条分支）→ summary.json → 报告页 / 复核页
原轨迹模型输出 ┘
                       ↑                  ↑
                 选点规则（按模型）   区域 + 跟随方式 + 模型适配器
```

1. plan：读原始评测轨迹、人工标注和原轨迹模型输出。偏离信息写进归档里原轨迹的 `rollout.jsonl`（见第五节），再按模型规则把偏离点换算成涂黑步和区域中心，写出 `candidates.jsonl`。
2. run：每个候选跑一条分支，从第 1 步起闭环运行。到涂黑步时定下区域，之后按跟随方式逐步涂黑，跑到结束后写 `run.json`。
3. analyze：读一个批次的全部 `run.json`，按 trajectory_id 成簇做 bootstrap，按组输出配对差和成功翻转计数，写 `summary.json`。
4. viz：根据 `run.json` 和 `summary.json` 生成报告页和人工复核页。

各阶段之间只通过文件交接，文件格式在 `core/store.py` 里定义。每个阶段可以单独重跑。批次的全部参数写进 `run.json`，批次名由参数生成。

## 三、模块

```
intervene/occlusion/
  core/                 只依赖 numpy，兼容 Python 3.6 / numpy 1.19
    geom.py             角度换算、四元数与朝向互转（两种分量顺序）、绕 y 轴旋转、两点间的世界朝向、折线距离、参考路径朝向、
                        方位弧的扫过与裁剪
    rig.py              相机配置：视图与光轴偏角、HFOV、RGB 与深度各自的分辨率、模型输入分辨率、深度单位与量程、相机高度；
                        不同分辨率之间的像素对应（最近邻）
    regions.py          区域定义：Heading（世界朝向区间）、Corridor
    trackers.py         跟随方式：Heading（锚点，可选判遮挡）、Surface、Volume；输出各视图的像素 mask，
                        并负责把 mask 作用到 RGB（及深度）上
    intervention.py     每步观测的钩子：判断这一步是否涂黑，调用区域与跟随方式，记下涂黑日志
    stats.py            分组 bootstrap、配对差、翻转计数
    store.py            candidates / run / summary 的字段定义与读写（原子写）、归档路径、出错记录、断点续跑
  agents/               每个模型一个适配器，在各自的 conda 环境里运行
    base.py             接口：episodes()、reset(ep)、observe(obs) → obs、step()、pose()、metrics()；
                        to_frame(obs, step) → Frame：从本模型的观测字典取出各视图的 RGB、深度（用 rig.to_metres 换成米）和当前位姿；
                        from_frame(frame, obs) → obs：把 frame 里涂黑后的 RGB（及 depth_raw）写回观测字典；
                        observe(obs) 即 from_frame(intervention.on_obs(to_frame(obs, step)), obs)
    omninav.py
    simplevln.py
    etpnav.py
  select/               选点规则，每个模型一个
    omninav.py          d 为前向步号；中心为 d 处最后一个预测路点的相对方位 th4 换成的世界朝向，读原轨迹模型输出
    simplevln.py        d 为路点号；推出转向段与前进方向 H，以及区域第一次进入视野的步 b
    etpnav.py           d 为规划次数；中心为所选 ghost 最先被预测出来的那次规划，从当时位置指向它的世界朝向
                        （需先跑一遍不涂黑的批次 none，记下每个 ghost 最先在哪次规划被预测出来，即 ghost 来历）
  cli.py                plan / run / analyze / viz；参数为模型、区域、跟随方式、跟随步数、偏移、涂黑通道
  viz/
```

### core 的类与函数

core 要兼容 Python 3.6：不用 `dataclasses` 和 `from __future__ import annotations`，记录类型用 `typing.NamedTuple` 或普通类。下面签名里的 `list[...]`、`X | None` 是说明用的写法，3.6 不支持，代码里写成 `typing.List`、`Optional` 等。

**统一约定**（下面各函数不再重复说明）：

- 角度：度，float64，左为正。朝向 0 指向世界 −z。"世界朝向"是相对世界的，"相对方位"是相对智能体朝向的。
- 坐标：世界坐标 (x, y, z)，单位米，y 朝上。"水平"指 xz 平面，xz 点用 [..., 2] 数组表示。
- 数组：形状写成 `[H, W]` 这类形式。深度在进入 core 时统一换成米（float64），没打到的记 `inf`。
- mask：`{视图名: bool [H, W]}`，分辨率为该视图的 `mask_hw`，即模型实际看到的 RGB 分辨率。
- 原轨迹的位姿：`rollout.jsonl` 里 OmniNav、SimpleVLN 第 k 行 step 的 pos、yaw 是第 k 步动作执行之后的位姿；第 k 步观测时的位姿取第 k−1 行，k = 1 时取 ready 行。ETPNav 的 decision 行记的是规划时的位姿，不用错位。select 和 `wrong_path` 取位姿都按这条。

几何参数不设默认值，作为必填参数由批次配置给出，并写进 `run.json`。这些参数包括过道半径 rho、高度范围 hmin / hmax、近处不涂的距离 r_near、错误折线长度 length、原地转身的判定阈值 still、锚点网的连接上限 gap_max 和 near_z、射线采样步长 ds、深度偏置 depth_bias 和补方向射线的间隔 unseen_step。只有方位带半宽有文献依据：取 15°，即 VLN-CE 的单步转角（Krantz et al. 2020）。

**geom.py**：纯函数，只用 numpy。

```python
wrap180(a: float | ndarray) -> float | ndarray
    # 换到 (−180°, 180°]，形状不变
yaw_from_quat(q: Sequence[float] (4,), order: str) -> float
    # 纯偏航四元数 → 世界朝向；order = 'wxyz'（OmniNav、SimpleVLN）| 'xyzw'（ETPNav）
quat_from_yaw(yaw: float, order: str) -> list[float] (4,)
rot_y(deg: float) -> ndarray [3, 3]
    # 绕 y 轴旋转；前方 (0, 0, −1) → (−sin deg, 0, −cos deg)
heading(p: Sequence[float] (3,), q: Sequence[float] (3,)) -> float
    # 从 p 指向 q 的世界朝向，只用 x、z
polyline_dist(q: ndarray [..., 2], path: ndarray [K, 2]) -> ndarray [...]
    # 点到折线的水平距离（米）；K = 1 时为到该点的距离
path_heading(pos: Sequence[float] (3,), ref: ndarray [M, 3], lam: float) -> float
    # 参考路径上，从离 pos 最近的点沿路径前进 lam 米处的世界朝向；超出终点取终点。只在 analyze / viz 里用
sweep(b: ndarray [N]) -> tuple[float, float]
    # b：按原顺序排好的一串相对方位。相邻差取 (−180°, 180°] 后累加。
    # 返回 (start, width)：start ∈ (−180°, 180°]，width ≥ 0；width ≥ 360 表示扫过整圈
clip_arcs(arcs: list[tuple[float, float]], lo: float, hi: float) -> list[list[float]]
    # arcs：(start, width)，方位相对智能体朝向。穿过 180° 的拆成两段，与视野 [lo, hi] 求交（lo、hi 同样相对智能体朝向，
    # 视图的视野为 [yaw_offset − hfov/2, yaw_offset + hfov/2]），按起点排序，重叠的合并。返回 [[起点, 终点], ...]。
    # 任一弧 width ≥ 360（扫过整圈）时覆盖整个视野，直接返回 [[lo, hi]]（C66、C67 相同）
```

**rig.py**：相机与单步观测。三个 Rig 实例的相机参数由 `from_config` 从模型自己的配置文件读出，不在代码里写死；`depth_bias` 由批次配置给出。

```python
class Pose(NamedTuple):
    position: ndarray (3,)     # 智能体位置（米）
    yaw: float                 # 世界朝向

class Frame:
    step: int                  # 步号，从 1 起；ETPNav 为规划次数
    pose: Pose
    rgb: dict[str, ndarray uint8 [H, W, 3]]        # 模型实际看到的分辨率（mask_hw）
    depth: dict[str, ndarray float64 [H, W]]       # 米，没打到为 inf；分辨率为 depth_hw；只给 tracker 读，适配器每步都给（见适配器一节的深度相机）
    depth_raw: dict[str, ndarray] | None            # 模型读的原始深度读数；apply 涂深度时写这里

class View(NamedTuple):
    name: str                  # 观测键，如 'left'、'front'、'rgb_30'
    yaw_offset: float          # 光轴相对智能体朝向的偏角
    hfov: float                # 水平视场
    depth_hw: tuple[int, int]  # 深度图分辨率
    mask_hw: tuple[int, int]   # 涂黑作用的分辨率，即模型看到的 RGB

class Rig:
    name: str                  # 'omninav' | 'simplevln' | 'etpnav'
    views: list[View]
    depth_scale: float         # 深度读数 × depth_scale = 米（归一化深度时等于量程）
    depth_max: float           # 深度量程（米）；读数与上限相差不到 1 个 float32 ulp 的记为没打到
    cam_height: float          # 相机离地高度（米）
    quat_order: str            # 'wxyz' | 'xyzw'
    depth_bias: float          # 判遮挡时深度比较的偏置（米），shadow map 需要它（Williams 1978）

    from_config(name: str, cfg: dict) -> Rig
    view(name: str) -> View
    to_metres(raw: ndarray [H, W] | [H, W, 1]) -> ndarray [H, W]
    column_heading(view: str, width: int) -> ndarray [W]
        # 各列中心的相对方位，已含 yaw_offset
    rays(view: str, hw: tuple[int, int], pose: Pose) -> tuple[ndarray [H, W, 3], ndarray (3,)]
        # 各像素中心射线的世界方向（相机系 z 深度为 1），以及光心 position + (0, cam_height, 0)
    unproject(view: str, depth: ndarray [H, W], pose: Pose) -> tuple[ndarray [H, W, 3], ndarray bool [H, W]]
        # 世界点与是否打到
    project(view: str, pts: ndarray [N, 3], is_dir: ndarray bool [N], pose: Pose) -> tuple[u [N], v [N], zc [N], inside bool [N]]
        # 投到 depth_hw 上：像素坐标（像素中心在 k + 0.5）、相机系深度、是否落在图内且在相机前方。
        # is_dir 为 True 的 pts 是世界方向：不减相机位置，只做旋转再投影（同 C67 Anchors._test）；zc 是方向在相机系的 z 分量，只用来判断在不在相机前方
    depth_ref(depth: ndarray [H, W]) -> ndarray [H, W]
        # 每个格子取 (R, C) 到 (R + 1, C + 1) 四角深度的最大值。右边和下边各补一行 / 一列（边缘复制）再取，所以输出仍是 [H, W]。
        # 没打到或到了量程上限（depth_max）的像素记 inf（输入是已换成米的 depth，没打到本来就是 inf）
    visible(view: str, pts: ndarray [N, 3], is_dir: ndarray bool [N], depth: ndarray [H, W], pose: Pose) -> ndarray bool [N]
        # is_dir 为 True 的只有方向，要求投到处也没打到；其余要求 zc ≤ depth_ref + depth_bias
    pixel_map(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> tuple[rows [H_dst], cols [W_dst]]
        # 最近邻对应，与 PIL.Image.NEAREST 一致
    nearest_view(rel: ndarray [N]) -> ndarray int64 [N]
        # 相对方位离哪个视图的光轴最近，返回 views 的下标
```

**regions.py**：区域在世界坐标里定义，可序列化。

```python
class Region:
    kind: str                  # 类属性：'heading' | 'corridor'
    to_json() -> dict
    from_json(d: dict) -> Region          # 静态方法，按 d['kind'] 分派

class Heading(Region):
    center: float              # 世界朝向
    half: float                # 半宽；方位带取 15°（Krantz et al. 2020）
    lo: float                  # center − half，不 wrap
    hi: float                  # center + half，不 wrap
    unwrap(h: ndarray [N]) -> ndarray [N]  # 把世界朝向接到 center 附近：center + wrap180(h − center)
    contains(h: ndarray [N]) -> ndarray bool [N]

class Corridor(Region):
    path: ndarray [K, 2]       # 错误折线（xz）
    floor_y: float             # 地面高度，取涂黑步的智能体 y
    rho: float                 # 离错误折线的水平距离上限（米）
    hmin: float                # 离地高度下限（米）
    hmax: float                # 离地高度上限（米）
    r_near: float              # 离当前位置的水平距离下限（米），脚下不涂
    inside(pts: ndarray [N, 3], here: ndarray (2,)) -> ndarray bool [N]
    reach(origin: ndarray (3,)) -> float   # 从光心看，这块空间的最远距离（米），射线采样到这里为止

wrong_path(positions: ndarray [T, 3], length: float, still: float) -> ndarray [K, 2]
    # positions：原轨迹从涂黑步观测时的位置起的各点（由 store.trajectory_from 给出，第 0 个点就是涂黑步的位置）。
    # 返回累计 length 米的折线（xz）。与上一个点相距不到 still 米的步（原地转身）不加点
```

**trackers.py**：跟随方式，统一输出像素 mask。

```python
class Tracker:
    kind: str                  # 类属性：'heading' | 'surface' | 'volume'
    rig: Rig
    region: Region | None      # init 之后才有
    init(region: Region, frame: Frame) -> None   # 涂黑步调用一次
    mask(frame: Frame) -> dict[str, ndarray bool [H, W]]
        # 输出在 mask_hw 上。rows, cols = pixel_map(depth_hw, mask_hw)：模型输入的每个像素取它最近邻缩放时来自的那个深度像素。
        # Volume 直接在 mask_hw 的像素上算，用那个深度像素的中心射线和深度（C66 的 model_grid、C69 的 rays）；
        # Surface 先在 depth_hw 上填格子，再取 m[rows][:, cols]；Heading 按 mask_hw 的宽度用 column_heading 算
    state() -> dict            # 写进 run.json 的摘要，各子类字段见下

# 与已有实验的对应：FOLLOW_V = 1（C66）= HeadingTracker(occlusion=False, rows='axis')；
# FOLLOW_V = 2（C66 后期、C67）= occlusion=True；FOLLOW_V = 3（C67 的网格）= SurfaceTracker。
class HeadingTracker(Tracker):   # 只接受 Heading 区域
    occlusion: bool            # True：被挡住的锚点不涂，连线在那里断开
    rows: str | int            # 'axis'：只取光轴高度一行（行数为偶数时取中间两行）；整数 n：每隔 n 行取一行，另加最后一行
    incremental: bool          # True：每步给新转进视野、还没定锚的朝向补锚（单视图用）；False：涂黑步一次定完
    unseen_step: float | None  # 从没进过视野的朝向，每隔这么多度补一条只有方向的射线；None 不补
    # 锚点，长度 N 的平行数组：
    a_heading: ndarray [N]     # 锚点取自的世界朝向（接在 center 附近）
    a_xz: ndarray [N, 2]       # 命中点（xz）；没打到的为定锚时的位置
    a_range: ndarray [N]       # 水平距离（米），没打到为 inf
    a_row: ndarray int64 [N]   # 取自第几行（0 起，按 rows 的顺序）
    a_w: ndarray [N, 3]        # 判遮挡用：命中点的世界坐标；没打到的为射线方向
    covered: list[list[float]] # 已定锚的朝向段 [a, b]
    _add(frame: Frame) -> int                    # 定锚或补锚，返回新增的锚点数
    _keep(frame: Frame) -> ndarray bool [N]      # 这一步参与涂黑的锚点：去掉正好站在上面的，occlusion 时再去掉被挡住的
    spans(frame: Frame, view: str) -> list[list[float]]
        # 这个视图里的涂黑区间，方位相对智能体朝向（与 column_heading 同一个零点，不是相对该视图光轴）。
        # 按 a_row 分行、按 a_heading 排序，在 _keep 为 False 处断开成几串，每串 sweep，
        # 再 clip_arcs 到 [yaw_offset − hfov/2, yaw_offset + hfov/2]
    # mask：column_heading 给出的列中心方位落在 spans 里的整列为 True（两者同为相对智能体朝向，直接比较）
    # state：n_anchor、n_hit、n_hidden、各视图 spans

class SurfaceTracker(HeadingTracker):   # 只接受 Heading 区域；定锚同 HeadingTracker，rows 取整数 n，总是判遮挡
    gap_max: float             # a_heading 相邻两条差过这么多度（中间没定锚）的不连
    near_z: float              # 相机系深度不到这么多（米）的角点算在相机后方
    # mask：锚点按（a_row, a_heading）连成网，相邻四个锚点围成一格，四个角都看得见的格子投到这一步的图上填满。
    #       投影时 is_dir = ~isfinite(a_range)：没打到的锚点（a_w 是射线方向）按方向投影，和打到的角点一起围成格子（同 C67）。
    #       投不进图但在相机前方的角点算看得见，由图像边界截掉；有角点在相机后方的格子不涂
    # state：同 HeadingTracker，另加 n_cell（这一步填了的格子数）

class VolumeTracker(Tracker):    # 接受 Corridor
    ds: float                  # 沿射线的采样步长（米）
    # mask：从光心起每隔 ds 采样到 min(命中距离, reach)，有采样点落进区域的像素，另加命中点本身。
    #       沿射线一步一步走（C69 的 Volume.mask）：每一步只算还没涂、也没越过命中点的像素，内存只有一个 [H, W] 的 bool，
    #       不展开 [H, W, S, 3] 的全部采样点
    # state：n_sample_max

apply(frame: Frame, masks: dict[str, ndarray bool [H, W]], channels: tuple[str, ...]) -> tuple[Frame, dict]
    # channels ⊆ ('rgb', 'depth')。RGB 置 0；深度先用 rows, cols = pixel_map(mask_hw, depth_hw) 把 mask 取到 depth_hw 上，再置 0。
    # 第二个返回值：{视图: {'pixels': int, 'cols': int}}
make_tracker(spec: dict, rig: Rig) -> Tracker
    # spec = {'kind': ..., 其余为该类的参数}
```

**intervention.py**：适配器每步观测进来时调用这里。

```python
class Intervention:
    cand: dict                 # candidates.jsonl 的一行
    rig: Rig
    region_spec: dict          # {'kind': ..., 其余为区域参数}；Heading 的 center 取自 cand['center']
    tracker: Tracker           # make_tracker(tracker_spec, rig) 构造
    channels: tuple[str, ...]  # 涂哪些通道，见 apply
    start: int                 # 涂黑步，取 cand['step']
    end: int | None            # 最后一个涂黑步 = start + follow；follow 为 None 时跟到结束，end 为 None
    region: Region | None      # 到 start 才构造
    records: list[dict]        # 每个涂黑步一条

    orig_positions: ndarray [T, 3] | None   # 原轨迹从涂黑步起的位置，store.trajectory_from 读出；只有 Corridor 区域需要，其余传 None
    # 回放（SimpleVLN）保证分支在涂黑步与原轨迹同位，所以折线可以直接取自原轨迹

    __init__(cand: dict, region_spec: dict, tracker_spec: dict, follow: int | None,
             channels: tuple[str, ...], rig: Rig, orig_positions: ndarray | None)
    active(step: int) -> bool                  # start ≤ step 且（end 为 None 或 step ≤ end）
    build_region(frame: Frame) -> Region       # 按 region_spec 和这一步的观测构造；Corridor 的折线由 wrong_path(orig_positions, ...) 得到，floor_y 取 frame.pose.position[1]
    on_obs(frame: Frame) -> Frame
        # 不在 active 范围内时原样返回。到 start 时先 build_region 和 tracker.init；
        # 从 start 起每步 tracker.mask → apply，返回模型看到的 Frame
    log() -> list[dict]
        # 每条：step、position（3 个数）、yaw、views（apply 的统计）、tracker（tracker.state()）
```

**stats.py**：SR、SPL（Anderson et al. 2018）和 nDTW（Ilharco et al. 2019）由各模型的评测代码计算，这里只读记录里的值。

```python
N_BOOT = 1000                  # bootstrap 次数，百分位区间所需的次数（Efron & Tibshirani 1993）
LEVEL = 0.95                   # 置信水平

cluster_weights(clusters: ndarray str [N], n: int, seed: int) -> ndarray float64 [n, N]
    # 按簇重抽样（cluster bootstrap，Davison & Hinkley 1997）：每次有放回地抽与簇数相同的簇，
    # 权重为该行所在簇被抽中的次数。簇取 trajectory_id，同一路径的几条指令一起抽
mean_ci(v: ndarray [N], W: ndarray [n, N], mask: ndarray bool [N] | None, level: float) -> list
    # [均值, 下限, 上限, 有效条数]；NaN 与 mask 外的不计
paired(rows: list[dict], metrics: tuple[str, ...], W: ndarray [n, N],
       groups: dict[str, ndarray bool [N]]) -> dict
    # {组名: {'n': int, 指标: {'orig': mean_ci, 'branch': mean_ci, 'delta': mean_ci}}}，delta = branch − orig
flips(rows: list[dict]) -> dict[str, int]
    # 键为 '0->0'、'0->1'、'1->0'、'1->1'，按 orig 与 branch 的 success
```

**store.py**：文件格式与归档读写，布局见第五节。

```python
ARCHIVE = Path('/media/jwm/大数据/result')
SERIES = {'omninav': 'C6002o', 'simplevln': 'C6002s', 'etpnav': 'C6002e'}
D_UNIT = {'omninav': 'forward_step', 'simplevln': 'waypoint', 'etpnav': 'planning'}

CANDIDATE = {                  # candidates.jsonl 每行；值为类型
    'model': str, 'split': str, 'ep': str, 'scene': str, 'trajectory_id': str,
    'd': int, 'd_unit': str,   # 人工偏离点及其单位（D_UNIT）
    'offset': int, 'step': int,                # 涂黑步 = 按模型规则由 d + offset 换算
    'center': (float, None),   # 区域中心（世界朝向）；ETPNav 在不涂黑批次记下 ghost 来历之前为 None
    'category': str, 'status': str, 'review_file': str, 'source_sha256': str,
    'rule_version': str, 'extra': dict}       # extra：各模型选点规则的中间量（如 SimpleVLN 的 s、c、c*、e、θ、b）
RUN = {                        # run.json
    'batch': str, 'params': dict,             # params：region_spec、tracker_spec、follow、channels、rig、offset
    'cand': dict,
    'orig': dict, 'branch': dict,             # 各模型评测给出的指标
    'blackout': list,                         # Intervention.log()
    'trajectory': list,                       # 分支每步的位置与朝向
    'seconds': float, 'at': str, 'env': dict} # env：Python 版本、git commit、conda 环境名
SUMMARY = {
    'batch': str, 'params': dict, 'n_boot': int, 'level': float, 'seed': int,
    'checks': dict,            # 记录数、出错数及类型、截断数
    'groups': dict, 'flips': dict}

validate(rec: dict, schema: dict) -> None                  # 缺字段或类型不符时抛 ValueError
batch_name(params: dict) -> str
    # '<区域>-<跟随方式>-<跟随步数>-<8 位十六进制>'，十六进制取 params 规范化 JSON 的 sha256 前 8 位
trajectory_from(model: str, split: str, ep: str, step: int) -> ndarray [T, 3]
    # 原轨迹从第 step 步观测时的位置起的各点，读 primary/rollout.jsonl。
    # OmniNav、SimpleVLN：第 step−1 行（step = 1 取 ready 行）起的 pos，位姿错位见统一约定。
    # ETPNav：第 step 次规划的位置所对应的位置记录起，之后各 step 行的 position；规划次数到位置记录序号的对应由 decision 行给出
episode_dir(split: str, ep: str) -> Path                   # ARCHIVE/<split>/<split>-<ep>
run_path(model: str, split: str, ep: str, step: int, batch: str) -> Path
    # episode_dir/<model>/intervene/<SERIES>/<step:04d>/<batch>/run.json
experiment_dir(model: str) -> Path                         # ARCHIVE/experiments/<model>/intervene/<SERIES>
read_jsonl(path: Path) -> list[dict]
append_jsonl(path: Path, rec: dict) -> None
write_json_atomic(path: Path, obj: dict) -> None           # 先写 <name>.tmp 再改名
load_reviews(path: Path, sha_of: Callable[[str], str]) -> dict[str, dict]
    # {ep: 标注}；sha_of(ep) 由各模型的 select 模块提供，与标注里的 source_sha256 不同时抛错
append_deviation(split: str, ep: str, model: str, row: dict) -> None
    # 在 <model>/primary/rollout.jsonl 末尾追加 {'event': 'deviation', ...}；已有 deviation 行时抛错
update_deviation(split: str, ep: str, model: str, fields: dict) -> None
    # 改写已有的 deviation 行（ETPNav 补 center），其余行逐字节不变
MAINT = ARCHIVE / 'maintenance' / 'c6002-<YYYYMMDD>'      # 本系列的维护目录，每次 commit 一个日期
record(model: str, op: str, path: Path, before: dict | None) -> None
    # 每写一个归档文件追加一行到 experiment_dir/records.jsonl：
    # {'op': 'write' | 'append' | 'rewrite', 'path': 相对 ARCHIVE, 'bytes', 'sha256', 'before': {'bytes', 'sha256'} | None, 'at'}
    # 运行期间只写这一份，不碰根目录元数据；三个环境并行时各自追加到各自模型的 experiment_dir
commit_meta(models: list[str]) -> None
    # 批次跑完后单独执行一次，单进程：
    # 1. 把 checksums.jsonl、manifest.json、README.md、settings.json 复制到 MAINT 下，
    #    命名为 checksums.before.jsonl、manifest.before.json、README.before.md、settings.before.json
    # 2. 合并各 records.jsonl，按 path 去重取最后一条，复制到 MAINT/records.jsonl
    # 3. checksums.jsonl：按 path 替换已有行，新文件追加行；行格式 {'path', 'bytes', 'sha256'}，path 相对 ARCHIVE
    # 4. manifest.json：episodes[].files 加入新文件（相对 episode 目录）；counts 加
    #    'alternative run.json：<model>/intervene/<SERIES>'；experiments 加 experiment_dir 下的新文件；
    #    maintenance 加 MAINT 下的文件；顶层加 'c6002_<YYYYMMDD>': {date, records, backups}
    # 5. settings.json 补 deviation 行和 C6002o、C6002s、C6002e 的字段说明；README.md 的目录与历史各补一段
    # 6. 最后重算 README.md、settings.json、manifest.json 自己在 checksums.jsonl 里的行（checksums.jsonl 不含自身）
    # 每一步先写 <name>.tmp 再改名
done_keys(model: str, split: str, batch: str) -> set[tuple[str, int]]   # 已有 run.json 的 (ep, step)
append_error(model: str, batch: str, rec: dict) -> None    # 追加到 experiment_dir/errors.jsonl
```

### 接口约定

- **区域**（`regions.py`）：在世界坐标系里定义，和模型、相机无关。涂黑步构造一次，之后保持不变。
- **跟随方式**（`trackers.py`）：`init(region, frame)` 在涂黑步定锚，`mask(frame)` 每步给出各视图要涂黑的像素；`rig` 在构造 tracker 时传入。按朝向跟随也输出像素 mask，整列涂黑只是它的一种特例。
- **模型适配器**（`agents/`）：驱动环境和模型的循环留在适配器里，每步观测进来时调用 `intervention.py`。适配器不包含选点和区域的逻辑。
- **深度相机**：适配器在环境里为每个视图额外加一个深度相机，位置、朝向、视场与该视图的 RGB 相机相同，量程为 `depth_max`，不归一化，读数以米为单位（C66、C67 的做法）。它的读数放进 `Frame.depth`，只给 tracker 用，不喂给模型；加它不能改变 RGB，要在自测里核对。模型自己读的深度（ETPNav 的规划器和 waypoint 预测器）是另一路，放在 `depth_raw`，只有 `channels` 含 `depth` 时才涂。
- **选点规则**（`select/`）：只读原始轨迹、标注和原轨迹模型输出，不调用模型。方位带中心在 plan 阶段定下（ETPNav 要等不涂黑批次跑完，再由 plan 补上），写进 `candidates.jsonl`，run 直接读取。

### 三个模型的约束

- **OmniNav**：三视图，每步观测；路点回归，确定性，可逐步核对；omninav 环境（Python 3.9）；涂 RGB。
- **SimpleVLN**：单视图，每步观测；用 vLLM 采样生成动作序列，不能逐条复现；simplevln-eval 环境（Python 3.9）加 vLLM 服务；涂 RGB。涂黑步之前不采样，逐步回放原轨迹 `rollout.jsonl` 里记的 action，使分支在涂黑步的位置和朝向与原轨迹一致，select 算出的 H 和 b 才对得上；从涂黑步起才让模型生成。回放结束时核对分支位姿与原轨迹第 start−1 行（start = 1 时为 ready 行）一致，不一致写入出错记录。
- **ETPNav**：12 视图全景，只在规划时观测；在拓扑图上选 ghost，取 argmax，确定性，可逐步核对；etpnav 环境（Python 3.6）；只涂 RGB，或 RGB 和深度都涂；深度涂过后，规划器和 waypoint 预测器读到的都是涂过的深度（waypoint 预测器只读深度，只涂 RGB 时它的输出不变）。

SimpleVLN 涂黑步之后的生成带采样，同一条候选再跑一次结果可能不同。每条候选只跑一次，`run.json` 记这一次的结果。

## 四、阶段

1. **core 与离线区域**。不需要模型，用 habitat_sim 按原轨迹位姿渲染，导出各区域、各跟随方式的 mask 和统计。自测项目：
   - 原位姿下，HeadingTracker 还原出定锚时的区间。
   - 只转身不平移时，区间平移的角度等于转角。
   - 判遮挡时，被挡住的锚点不涂。
   - Surface 跟随在原位姿下与定锚时的 mask 一致。
   - 加了深度相机之后，渲染的 RGB 与不加时逐像素相同。
   - etpnav 环境（Python 3.6）下 core 能 import，自测能跑通。
2. **OmniNav**。适配器、plan、analyze 先在这个模型上打通，跑方位带加朝向跟随。
3. **ETPNav**。先跑不涂黑批次 none 记下 ghost 来历，再跑涂 RGB、涂 RGB 加深度两组。
4. **SimpleVLN**。跑方位带加朝向跟随，每条候选一次。
5. **组合实验**。三维区域（错误过道）接到三个模型上，与方位带对比。
6. **可视化**。报告页（分组指标、成功翻转）、复核页（逐条分支的俯视轨迹与涂黑图）、mask 可视化。

## 五、产物位置

偏离信息和分支记录写在大数据盘的归档 `/media/jwm/大数据/result`，沿用它现有的布局。系列名为 C6002o（OmniNav）、C6002s（SimpleVLN）、C6002e（ETPNav）。

```
<split>/<split>-<id>/
  omninav/primary/rollout.jsonl                  追加一行 event = deviation
  omninav/intervene/C6002o/<涂黑步>/<批次>/run.json
  simplevln/primary/rollout.jsonl                追加一行 event = deviation
  simplevln/intervene/C6002s/<涂黑步>/<批次>/run.json
  etpnav/primary/rollout.jsonl                   追加一行 event = deviation
  etpnav/intervene/C6002e/<涂黑步>/<批次>/run.json
experiments/<模型>/intervene/C6002<o|s|e>/       候选、批次参数、写入记录 records.jsonl、出错记录、日志
```

- **偏离信息**：在原轨迹 `rollout.jsonl` 末尾追加一行 `event = deviation`，原有的行不动。这一行包含：
  - 人工偏离点 d 及其单位：OmniNav 为前向步，SimpleVLN 为路点号，ETPNav 为规划次数。ETPNav 的 rollout 里 step 是位置记录序号，不等于规划次数，所以这一行另记 d 对应的位置记录序号。
  - 失败类别、标注状态、标注文件名、`source_sha256`。
  - 换算出的涂黑步和区域中心，以及换算规则的版本。ETPNav 的区域中心要等不涂黑批次 none 记下 ghost 来历后才能算，算出后补写进这一行。
- **分支记录**：每条分支一个 `run.json`。`<涂黑步>` 为 4 位、从 1 起，与归档其余目录一致。ETPNav 的涂黑步按规划次数编号。`<批次>` 由参数生成，不涂黑的 none 也是一个批次。
- **归档元数据**：`checksums.jsonl`、`manifest.json`、`settings.json`、`README.md` 都只在归档根目录，没有逐 episode 的副本。`checksums.jsonl` 每个文件一行 `{path, bytes, sha256}`；`manifest.json` 记 `episodes[].files`、`counts`、`experiments`、`maintenance` 和历次维护的顶层条目。改写 `rollout.jsonl` 和新增 `run.json` 都会让它们失效。做法沿用归档已有的维护惯例（`maintenance/<操作>-<日期>/` 下放 before 备份和 `records.jsonl`）：运行期间只追加 `records.jsonl`，批次跑完后由 `store.commit_meta` 单进程统一更新，见第三节 store.py。
- **工作区**：`intervene/results/` 只放 analyze 的 `summary.json` 和页面。人工标注只读；原始评测只在 `rollout.jsonl` 末尾追加 deviation 行。
