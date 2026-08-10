# EEG–握力团队数据接口

`grip_data_interface.py` 将两份同步的 BCI2000 记录整理成与
`playgroundgit` 的 `pen_split` 相同风格的变长 trial 字典。原始数据和生成结果
仍由 `.gitignore` 排除，团队共享的是接口代码，不是患者数据。

## 最小用法

```python
from pathlib import Path
from grip_data_interface import load_all_flight_trials, load_flight_trials

data_dir = Path("0807华山grip flight")
r11 = load_flight_trials(
    data_dir / "testS001R11.dat.larkcache",  # EEG
    data_dir / "testS001R11_1.dat",          # GripFlightTask
    split_name="all",
)

eeg_trial0 = r11["X_list"][0]        # (n_channels, n_samples)
force_trial0 = r11["target_list"][0] # (1, n_samples)
trial_table = r11["meta"]
event_table = r11["events"]
```

批量分析时使用 `load_all_flight_trials()`，一次读取后同时得到：

```python
splits = load_all_flight_trials(eeg_path, task_path)
splits["all"]
splits["success"]
splits["failure"]
splits["collision"]
```

## 与 playgroundgit 的字段对应

| playgroundgit `pen_split` | 当前 `flight_split` | 形状/含义 |
|---|---|---|
| `X_list` | `X_list` | 每个 trial 的 EEG，`channels × samples` |
| `mouse_xy_list` | `target_list` | 连续回归目标；当前为一行原始握力 |
| `meta` | `meta` | 每个 trial 一行的 DataFrame |
| `sr` | `sr` | EEG 采样率 |
| `time_list` | `time_list` | 每个 trial 从零开始的时间轴 |
| 通道信息 | `channel_names/channel_indices` | 有效通道名称和原始索引 |

额外提供：

- `force_list`：一维原始握力，方便绘图；
- `force_normalized_list`：任务中保存的归一化握力；
- `state_list`：与 EEG 采样点对齐的球位置、速度、Collision 等状态；
- `events`：全局语义事件表，不暴露容易混淆的原始 code；
- `event_list`：按 trial 分开的事件表；
- `source_time_ms_list`：每个 trial 的 SourceTime 相对时钟。

## Trial 范围

默认 `segment="flight"`：

- 从 `flight_start` 开始；
- 成功 trial 在结果画面前结束；
- 失败 trial 保留 Collision onset 样本；
- 不包含 Countdown、结果展示和 ITI。

它相当于 WritingStrokes 接口中的真实 pen-down segment。还可以选择：

- `segment="playing"`：严格保留 `GamePhase == Playing`；
- `segment="trial"`：保留 Countdown 至下一个 trial 前的完整区间。

## 建模约定

`target_list[i]` 与 `X_list[i]` 具有完全相同的时间长度。模型划分必须按
`meta.trial_id` 或整段 trial 进行，不要随机拆分相邻采样点。球位置、速度和
Collision 是握力经过游戏物理后的下游量；如果研究目标是 EEG→握力，不应将
它们作为输入特征。

接口版本保存在 `data["interface_version"]`。后续修改字段语义时应同步升级版本。
