# 实机测试说明（探针版）

探针版用 `.\LandingAssist\pack.ps1 -Probe` 构建，版本号带 `-probe` 后缀，探针和自动录制默认开启。验收项目见 [发布前清单](release-checklist.md)。

## 设置

配置文件是游戏根目录下的 `UserData/MelonPreferences.cfg`。

- `[MaydayLandingAuto]`
  - `ExperimentalFlightTests`：是否允许接管，默认 `true`。
  - `AutoMode`：启动时是否进入全自动，默认 `false`。
- `[MaydayLandingProbe]`
  - `Probe`：记录原生阵风、接地、危险事件，以及模型与 prior 的对比。探针版默认 `true`，正式版默认 `false`。
  - `AutoRecord`：每次进近自动录制，停稳或离开进近 20 s 后结束。默认值与 `Probe` 相同。

从探针版换回正式版时，要手动把这两项改回 `false`，因为已经写进配置文件的值会保留。如果游戏在接地或天气事件时报错，先把 `Probe` 设为 `false`，这会关闭 Harmony 钩子。

## 按键

| 键 | 作用 |
|---|---|
| F7 | 建议 ↔ 全自动；接管中按下即断开 |
| F5 | 复飞 |
| F6 | 手动开始或停止录制；手动停止后，本局不再自动录制 |
| Shift+F6 | 标记：分析时会优先查看该时刻前后 5 s |

## 建议的测试组合（房主）

1. 晴天手飞 2–3 趟：对比模型建议和实际操作。
2. 大风天气（gusts 或 low_vis_windy）手飞 2–3 趟：核对阵风冲击和接地损伤。
3. 晴天全自动。
4. 大风天气全自动。

复飞、交还和中断的航次也要保留，它们同样是有用的数据。

## 录制文件

保存在 `UserData/MaydayLanding/{auto,advisor}/rec/`，每趟两个文件：

- `landing-*.jsonl`：逐帧记录，包括观测、建议、授权的轴、提交的操纵和驾驶员杆量。
- `landing-*.summary.json`：本趟摘要，包括接通时长、交还原因、最大观测延迟、阵风次数、接地、危险事件和标记。

录制头里的 `policySha256` 标识当时实际使用的权重。

## 分析

```powershell
cd tools
uv run --with numpy python -m mayday_landing.probe_report <rec>\landing-*.jsonl --json probe.json
uv run --with numpy --with jax python -m mayday_landing.replay_trace <记录>.jsonl --model ..\LandingAssist\Models\candidate.json
uv run python -m mayday_landing.compare_traces <房主>.jsonl <客户端>.jsonl --same-flight
```

`probe_report` 会给出以下内容：

- 实测的阵风滚转冲击，与孪生的 12° 对比；
- 接地事件，以及按原生规则判定的坠机、HardLanding 和擦地；
- 危险事件；
- 观测延迟分位数；
- 模型与 prior、模型与驾驶员之间的平均差；
- 每个标记附近的片段。
