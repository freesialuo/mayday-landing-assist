<div align="center">

# Mayday Landing Assist

[![Version](https://img.shields.io/badge/version-0.2.0-blue?style=flat-square)](LandingAssist/LandingRuntime.props)
[![Status](https://img.shields.io/badge/status-experimental-orange?style=flat-square)](#status)
[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-lightgrey?style=flat-square)](LICENSE)
[![Game](https://img.shields.io/badge/game-Mayday%20Airways-1f6feb?style=flat-square)](#)
[![MelonLoader](https://img.shields.io/badge/MelonLoader-Il2Cpp%20net6-8a2be2?style=flat-square)](https://github.com/LavaGang/MelonLoader)
[![.NET](https://img.shields.io/badge/.NET-6.0-512BD4?style=flat-square&logo=dotnet&logoColor=white)](https://dotnet.microsoft.com/)
[![Twin](https://img.shields.io/badge/twin-JAX-FF6F00?style=flat-square)](tools/mayday_jax)

**English** · [简体中文](README.zh-CN.md)

</div>

An experimental approach and landing assistant for Mayday Airways. Advisor mode shows hand-flying cues in a native-style panel under the aircraft card in the top-left corner. Full Auto mode takes over on final, flies through the five rings, flares, touches down, uses reverse thrust, brakes to a stop and hands control back. Any stick input returns control to the pilot immediately.

> License: [PolyForm Noncommercial 1.0.0](LICENSE) (SPDX: `PolyForm-Noncommercial-1.0.0`). Noncommercial use only; redistributions and derivatives must keep the `Required Notice` lines. Source files carry an `SPDX-License-Identifier` header.

## Features

| Feature | Description |
|---|---|
| Persistent panel | Built with the game's own `UIF` factory, `UITheme` colours and rounded font; docks under the native aircraft card and fades in and out with the HUD. Falls back to an IMGUI panel if the native UI is unavailable |
| F7: Advisor ↔ Full Auto | Full Auto arms itself in the air and takes over once inside the engagement window. Stick input, F7 or a fault hands control back and returns to Advisor mode |
| Engagement window | 4.75–8.5 km from the threshold is the wide window, 1.05–4.75 km the strict one; outside the window the panel lists which conditions are still unmet |
| Native AP | If the native autopilot is flying when the window is reached, the AP is disengaged before takeover |
| Landing gear | The assistant never operates the gear, it only prompts you to lower it or hand-crank it. If the gear is still not locked at 65 m AGL it goes around once; if it is still not locked after that it hands control back |
| Rollout | Reverse thrust is used when both engines are healthy and ground speed > 10 m/s; reversers are always stowed before handing back; the parking brake is set once stopped |
| Recording & probe | F6 records per-frame JSONL; the probe build additionally logs native gusts, touchdown and hazard events for calibrating the twin |

Keys: F7 switch mode, F5 go around, F6 record, Shift+F6 add a marker to the recording (F8 and F9 are left free for other flight-control mods).

Two packages:

- **Auto**: cues plus experimental takeover.
- **Advisor**: cues only, does not modify the game; F7 only hides the panel.

Install only one of them.

## Repository layout

```
LandingAssist/
  Core/       Game-agnostic logic: director (authority, phases, handback), features, policy inference, prior, recording, probe
  Game/       Game integration: state reading, control submission, native UI panel, IMGUI fallback, Harmony hooks
  Advisor/ Auto/   Projects for the two packages (sharing LandingRuntime.props)
  Models/     Policy weights embedded in releases, candidate.json (inference fields only)
  Tests/      C# tests: director, prior parity, model, probe, package checks
  pack.ps1    Runs tests, builds and produces the .maydaymod
tools/
  mayday_jax/      Flight twin in JAX (physics, environment, replay)
  mayday_landing/  Landing environment, prior, teacher and student training, evaluation taxonomy, recording analysis
  tests/           Python tests
docs/                     (in Chinese)
  mechanics.md          Mechanics and constants confirmed from the game
  training-overview.md  Training method overview
  testing.md            Probe build and in-game testing
  release-checklist.md  Pre-release checklist
  roadmap.md            Further research plans
  nexus-release.md      Nexus Mods release text
```

## Building

Requirements:

- .NET SDK 6 or later;
- a copy of the game with MelonLoader installed and the Il2Cpp interop assemblies generated.

The repository is assumed to live at `<game dir>/ModSource/<repo>/`. If it lives elsewhere, pass `GameDir`:

```powershell
.\LandingAssist\pack.ps1                 # Release build: name carries "(Experimental)", probe off
.\LandingAssist\pack.ps1 -Probe          # Test build: probe and auto-recording on by default
.\LandingAssist\pack.ps1 -Version 0.2.1  # Set the version number
dotnet build LandingAssist\Auto\MaydayLandingAuto.csproj -c Release -p:GameDir="D:\Games\Mayday Airways"
```

Output goes to `dist/landing/`. To install, import the `.maydaymod` in the game's Mod Manager.

## Tests and analysis

```powershell
dotnet run -c Release --project LandingAssist\Tests\LandingTests.csproj
cd tools
uv run --with jax --with numpy --with optax --with pytest python -m pytest -q tests
uv run --with numpy python -m mayday_landing.probe_report <recording>.jsonl
uv run --with numpy --with jax python -m mayday_landing.replay_trace <recording>.jsonl --model ..\LandingAssist\Models\candidate.json
```

Python dependencies are pulled in on demand with `uv run --with`; nothing needs to be installed globally.

## Design notes

- **Control authority lives outside the model.** `LandingDirector` decides whether to write, which axes to write and when to hand back. Pilot input, stale data and configuration problems all trigger an immediate handback; the policy only proposes values.
- **Policy = interpretable prior + bounded residual.** The prior is a guidance law based on runway geometry and energy and can fly the whole approach on its own; the network only corrects it within limited bounds. If the network is missing or misbehaves, the assistant falls back to the prior or to cues only.
- **The twin is checked against the game.** Offline training and evaluation use a JAX twin; key mechanics (gear jams and hand-cranking, gust roll upsets, touchdown detection) are confirmed against game code and in-game recordings, see [mechanics.md](docs/mechanics.md).

The training method is described in the [training overview](docs/training-overview.md).

## Status

Experimental. Verified in-game only as host or in single player. In offline evaluation, judged by the game's own rules, there are no crashes; the remaining failures are occasional wingtip strikes and hard touchdowns on high-wind days. Two-player co-op (client flying) has not been tested. Panel text is currently Chinese only.
