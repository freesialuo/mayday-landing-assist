# Mayday Landing Assist

Mayday Airways 的进近与着陆辅助模组（实验性）。建议模式在左上角飞机卡片下方给出原生风格的手飞提示；全自动模式在五边接管，穿过五环、拉平、接地、开反推并刹停，之后交还操纵。玩家随时动杆即可收回操纵。

An experimental approach and landing assistant for Mayday Airways. Advisor mode shows hand-flying cues in a native-style panel under the aircraft card. Full Auto mode takes over on final, flies the five rings, flares, touches down, uses reverse thrust, brakes to a stop and hands control back. Any stick input returns control to the pilot immediately.

> 许可：[PolyForm Noncommercial 1.0.0](LICENSE)。只允许非商业使用，再分发或修改时必须保留 `Required Notice` 作者声明。
> License: PolyForm Noncommercial 1.0.0. Noncommercial use only; redistributions and derivatives must keep the `Required Notice` lines.

## 功能

| 功能 | 说明 |
|---|---|
| 常驻面板 | 用游戏自带的 `UIF` 工厂、`UITheme` 配色和圆体字搭建，停靠在原生飞机卡片下方，随 HUD 一起淡入淡出；拿不到原生 UI 时退回 IMGUI 面板 |
| F7：建议 ↔ 全自动 | 全自动模式在空中自动预位，进入接通窗口后接管。动杆、按 F7 或出现故障时交还并回到建议模式 |
| 接通窗口 | 距入口 4.75–8.5 km 为较宽窗口，1.05–4.75 km 为较严窗口；窗口外会列出还差哪些条件 |
| 原生 AP | 进入窗口时如果原生自动驾驶在飞，先断开 AP 再接管 |
| 起落架 | 辅助不操作起落架，只提示放下或手摇；离地 65 m 仍未锁定就复飞一次，之后仍未锁定则交还 |
| 滑跑 | 双发正常且地速 > 10 m/s 时开反推；交还前一定先收起反推；停稳后拉驻车制动 |
| 录制与探针 | F6 录制逐帧 JSONL；探针版额外记录原生阵风、接地和危险事件，用于校准孪生 |

按键：F7 切换模式，F5 复飞，F6 录制，Shift+F6 在录制里打标记（F8、F9 留给其他飞控模组）。

两个安装包：

- **Auto**：提示和实验接管。
- **Advisor**：只给提示，不修改游戏，F7 只用于隐藏面板。

只装其中一个。

## 仓库结构

```
LandingAssist/
  Core/       与游戏无关的逻辑：指挥器（授权、阶段、交还）、特征、策略推理、prior、录制、探针
  Game/       游戏接入：读取状态、提交操纵、原生 UI 面板、IMGUI 退路、Harmony 钩子
  Advisor/ Auto/   两个安装包的工程（共用 LandingRuntime.props）
  Models/     发布内嵌的策略权重 candidate.json（只含推理所需字段）
  Tests/      C# 测试：指挥器、prior 一致性、模型、探针、安装包检查
  pack.ps1    运行测试、构建并生成 .maydaymod
tools/
  mayday_jax/      JAX 实现的飞行孪生（物理、环境、回放）
  mayday_landing/  着陆环境、prior、教师与学生训练、评估分类、录制分析
  tests/           Python 测试
docs/
  mechanics.md          从游戏中确认的机制与常数
  training-overview.md  训练方法概述
  testing.md            探针版与实机测试
  release-checklist.md  发布前清单
  roadmap.md            后续研究计划
  nexus-release.md      Nexus Mods 发布文本
```

## 构建

需要：

- .NET SDK 6 或更高版本；
- 一份已安装 MelonLoader 并生成过 Il2Cpp 互操作程序集的游戏。

默认假设仓库位于 `<游戏目录>/ModSource/<仓库>/`。放在其他位置时传入 `GameDir`：

```powershell
.\LandingAssist\pack.ps1                 # 正式版：名称带 "(Experimental)"，探针关闭
.\LandingAssist\pack.ps1 -Probe          # 测试版：探针与自动录制默认开启
.\LandingAssist\pack.ps1 -Version 0.2.1  # 指定版本号
dotnet build LandingAssist\Auto\MaydayLandingAuto.csproj -c Release -p:GameDir="D:\Games\Mayday Airways"
```

产物写入 `dist/landing/`。安装：在游戏的 Mod Manager 里导入 `.maydaymod`。

## 测试与分析

```powershell
dotnet run -c Release --project LandingAssist\Tests\LandingTests.csproj
cd tools
uv run --with jax --with numpy --with optax --with pytest python -m pytest -q tests
uv run --with numpy python -m mayday_landing.probe_report <录制>.jsonl
uv run --with numpy --with jax python -m mayday_landing.replay_trace <录制>.jsonl --model ..\LandingAssist\Models\candidate.json
```

Python 依赖按需用 `uv run --with` 引入，不需要全局安装。

## 设计要点

- **操纵权在模型之外。** `LandingDirector` 决定能不能写、写哪几个轴、何时交还。驾驶员输入、数据过期和构型问题都会立即交还；策略只给建议值。
- **策略 = 可解释的 prior + 有界残差。** prior 是基于跑道几何和能量的引导律，可以单独飞完整个进近；网络只在限定幅度内修正它。网络缺失或异常时，退回 prior 或只给提示。
- **孪生对照游戏。** 离线训练和评估使用 JAX 孪生，关键机制（起落架卡滞与手摇、阵风滚转冲击、接地判定）对照游戏代码和实机录制确认，见 [mechanics.md](docs/mechanics.md)。

训练方法见 [训练概述](docs/training-overview.md)。

## 状态

实验性。只在房主或单人模式下实机验证过。离线评估里按游戏原生判定没有坠机，剩余失败是大风天偶发的翼尖擦地和接地偏重。双人联机（客户端驾驶）尚未测试。面板文字目前只有中文。
