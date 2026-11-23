# Nexus Mods 发布文本（0.2.0）

以下三栏可直接粘贴到 Nexus 上传页。Full description 使用 BBCode，英文在前、中文在后。

---

## Mod Name

```
Mayday Landing Assist (Experimental)
```

## Short description

（Nexus 限 350 字符，下面英文版约 280 字符。）

```
Approach and landing assist for Mayday Airways. Advisor mode shows game-style cues under the aircraft card; Full Auto can take over on final, fly the five rings, flare, touch down, use reverse thrust and brake to a stop. Move the stick at any time to take back control.
```

中文备用（如果想放双语）：

```
Mayday Airways 进近着陆辅助。建议模式在飞机卡片下方给出原生风格提示；全自动模式可在五边接管，穿环、拉平、接地、反推并刹停。随时动杆即可收回操纵。
```

## Full description（BBCode）

```bbcode
[size=5][b]Mayday Landing Assist[/b][/size]
[i]Experimental approach and landing assistant for Mayday Airways.[/i]

[size=4][b]What it does[/b][/size]
[list]
[*][b]Native-style panel[/b] docked under the aircraft card (top left). It uses the game's own UI theme and rounded font and fades with the HUD. It shows mode, phase, the next instruction, ring / height / lateral offset / sink rate, and recording state.
[*][b]Advisor mode[/b] (default): hand-flying cues only. Nothing is written to the controls.
[*][b]Full Auto mode[/b]: the assist arms itself while airborne and takes over once you are inside the capture window. It then flies the five rings, flares, touches down, uses reverse thrust on the roll-out and brakes to a full stop, then hands control back.
[*][b]Works with the native autopilot[/b]: if the game AP is flying when the window is reached, the assist disconnects it and takes over.
[*][b]Safe hand-back[/b]: moving the stick, pressing F7, stale data or a failure returns control to you immediately and drops back to Advisor. Reverse thrust is always stowed before hand-back.
[/list]

[size=4][b]Capture window[/b][/size]
[list]
[*]4.75–8.5 km from the runway: lateral offset up to 600 m (tighter closer in), ±150 m from the glide path, 35–70 m/s, track within 45°, bank within 25°.
[*]1.05–4.75 km: a tighter window.
[*]Outside the window the panel lists what is still missing (distance, offset, speed…), so you can fly into it.
[/list]

[size=4][b]What it does NOT do[/b][/size]
[list]
[*]It does not lower the gear or set flaps. If the gear is up or jammed, it keeps reminding you (lower it or hand-crank it). If the gear is still not locked at 65 m it goes around once; if it is still not locked after that, it hands control back. It never lands gear-up.
[*]No take-off, cruise or taxi.
[/list]

[size=4][b]Controls[/b][/size]
[list]
[*][b]F7[/b] – switch Advisor / Full Auto (while engaged: disconnect)
[*][b]F5[/b] – go around
[*][b]F6[/b] – start/stop recording a flight log
[*][b]Shift+F6[/b] – drop a marker into the recording
[*][b]Stick input[/b] – take over instantly
[/list]

[size=4][b]Installation[/b][/size]
[list=1]
[*]Download [b]one[/b] package: [b]mayday-landing-auto[/b] (recommended) or [b]mayday-landing-advisor[/b] (cues only, never touches the controls).
[*]Import the [i].maydaymod[/i] file in the Mod Manager. Do not install both.
[*]Start a flight and sit in the pilot seat. The panel appears under the aircraft card.
[/list]
Settings live in [i]UserData/MelonPreferences.cfg[/i] under [i][MaydayLandingAuto][/i]: [i]ExperimentalFlightTests[/i] (allow take-over, default true) and [i]AutoMode[/i] (start in Full Auto, default false).

[size=4][b]How it was built[/b][/size]
The controller is a guidance law plus a small neural residual, trained in a JAX digital twin of the game's flight model. Game mechanics (gear jam and crank, front gusts, touchdown crash / hard-landing rules) were checked against the game code and in-game recordings. In the twin, no evaluation produced a native crash; the remaining failures are occasional wing-tip strikes or firm touchdowns, mostly in gusty weather.

[size=4][b]Known limitations[/b][/size]
[list]
[*][color=#f2b84b]Experimental.[/color] Verified as host / single player only. Multiplayer with the client flying is untested.
[*]The panel text is currently [b]Chinese only[/b]. English UI is planned.
[*]Gusty weather can still cause an occasional wing-tip strike (damage, not a crash).
[*]Arming from fast cruise configuration (high speed, gear up) is less tested.
[*]A game update can break it. The panel falls back to a simple style; other features may switch off.
[*]Do not use together with another flight-control mod; if one is detected, this mod goes read-only.
[/list]

[size=4][b]Reporting problems[/b][/size]
Please attach the MelonLoader log and an F6 recording ([i]UserData/MaydayLanding/auto/rec/[/i]). Press Shift+F6 right when something looks wrong — it makes the moment easy to find.

[line]

[size=5][b]Mayday 着陆辅助（中文）[/b][/size]
[i]Mayday Airways 实验性进近着陆辅助。[/i]

[size=4][b]功能[/b][/size]
[list]
[*][b]原生风格面板[/b]：位于左上角飞机卡片下方，使用游戏自带的 UI 配色和圆体字，随 HUD 一起淡入淡出。显示模式、阶段、下一步指令、环 / 离地 / 侧偏 / 下沉率和录制状态。
[*][b]建议模式[/b]（默认）：只给手飞提示，不操作飞机。
[*][b]全自动模式[/b]：空中自动预位，进入接通窗口后接管；穿过五环、拉平、接地，滑跑时开反推并刹停，停稳后交还操纵。
[*][b]兼容原生自动驾驶[/b]：进入窗口时如果原生 AP 在飞，会先断开 AP 再接管。
[*][b]安全交还[/b]：动杆、按 F7、数据过期或出现故障时立即交还并回到建议模式；交还前一定先收起反推。
[/list]

[size=4][b]不会做的事[/b][/size]
[list]
[*]不放起落架、不调襟翼。起落架未放或卡滞时持续提示（放下或手摇）；离地 65 m 仍未锁定会复飞一次，复飞后仍未锁定则交还，绝不会带着收起的起落架落地。
[*]不负责起飞、巡航和滑行。
[/list]

[size=4][b]按键[/b][/size]
[list]
[*][b]F7[/b]：建议 / 全自动切换（接管中按下即断开）
[*][b]F5[/b]：复飞
[*][b]F6[/b]：开始 / 停止录制
[*][b]Shift+F6[/b]：在录制中打标记
[*][b]动杆[/b]：立即收回操纵
[/list]

[size=4][b]安装[/b][/size]
[list=1]
[*]只下载[b]一个[/b]包：[b]mayday-landing-auto[/b]（推荐）或 [b]mayday-landing-advisor[/b]（只给提示，从不操作飞机）。
[*]在 Mod Manager 里导入 [i].maydaymod[/i] 文件，不要两个都装。
[*]开始飞行并坐上驾驶座，面板会出现在飞机卡片下方。
[/list]

[size=4][b]已知限制[/b][/size]
[list]
[*]实验性功能，只在房主 / 单人模式下验证过，客户端驾驶的联机尚未测试。
[*]面板目前只有中文。
[*]大风天气偶尔会翼尖擦地（造成损伤，不会坠机）。
[*]从高速巡航构型（速度快、起落架收起）预位的情况测试较少。
[*]游戏更新后可能失效：面板会退回简易样式，其他功能可能停用。
[*]不要和其他飞控模组同时使用；检测到时本模组会进入只读。
[/list]

[size=4][b]反馈问题[/b][/size]
请附上 MelonLoader 日志和 F6 录制（[i]UserData/MaydayLanding/auto/rec/[/i]）。发现异常时按 Shift+F6 打标记，方便定位。
```

---

## 上传时的其他字段（建议）

- **Category**：Gameplay 或 Utilities。
- **Version**：0.2.0。
- **Files**：Main file 上传 Auto 包，Optional file 上传 Advisor 包；文件说明里写 SHA-256（见 [发布前清单](release-checklist.md) §0）。
- **Tags**：Flight、Quality of Life、Automation。
- **Requirements**：游戏的 Mod Manager / MelonLoader（以你实际的加载方式为准）。
- 首页截图：面板在建议模式、全自动接管中、滑跑反推各一张。
