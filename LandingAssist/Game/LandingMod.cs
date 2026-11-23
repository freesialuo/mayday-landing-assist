// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using Il2CppMayday;
using Il2CppMayday.Flight;
using MelonLoader;
using MelonLoader.Utils;
using UnityEngine;
using UnityEngine.InputSystem;

[assembly: MelonInfo(typeof(MaydayLanding.LandingMod), MaydayLanding.LandingMod.Name, "0.2.0", "MaydayLanding")]
[assembly: MelonGame("Lyngo Games", "Mayday Airways")]

namespace MaydayLanding;

public sealed class LandingMod : MelonMod
{
    public const string AutoName = "Mayday Landing Auto";
#if LANDING_AUTO
    public const string Name = AutoName;
    private const string Mode = "auto";
    private const double OutputExpiry = 0.2;
    private const double ActuationAgeLimit = 0.3;
    private MelonPreferences_Entry<bool>? _experimentalTests;
    private bool _patched;
    private bool _patchConflict;
    private double _checkPatchesAt;
    private double _lastThrottleTime = double.NegativeInfinity;
    private int _lastThrottle = -1;
    private double _lastParkTime = double.NegativeInfinity;
    private const float ReverseThrottle = 0.8f;
    private double _lastReverseTime = double.NegativeInfinity;
    private bool _reverseOwned;
    private double _lastApDisconnect = double.NegativeInfinity;
#else
    public const string Name = "Mayday Landing Advisor";
    private const string Mode = "advisor";
#endif
    private const double PolicyPeriod = 0.1;
    private const double ObservationExpiry = 0.6;
#if LANDING_PROBE
    private const bool ProbeDefault = true;
#else
    private const bool ProbeDefault = false;
#endif
    private const double AutoRecordLinger = 20.0;
    private MelonPreferences_Entry<bool>? _probeEnabled;
    private MelonPreferences_Entry<bool>? _autoRecord;
    private LandingProbe? _probe;
    private float[]? _priorAction;
    private bool _autoTrace;
    private bool _autoSuppressed;
    private double _outsideSince = double.NaN;
    private readonly LandingGame _game = new();
    private readonly LandingDirector _director = new();
    private readonly ControllerConflict _conflict = new();
    private readonly MotionResponse _response = new();
    private LandingPolicy? _policy;
    private string? _policySha256;
    private LandingTrace? _trace;
    private LandingFrame? _frame;
    private LandingDecision _decision;
    private float[]? _proposal;
    private float[]? _previousIntent;
    private LandingAxes _previousAxes;
    private double _proposalAt = double.NegativeInfinity;
    private bool _policyResetPending;
    private bool _hudEnabled = true;
    // Full auto: stay armed every approach and engage in the capture window. Advice: hints only.
    private bool _autoMode;
    private bool _armPulse;
    private MelonPreferences_Entry<bool>? _autoModeEntry;
    private bool _stopPending;
    private bool _updateError;
#if !LANDING_AUTO
    private bool _duplicateAdvisor;
#endif
    private string? _traceError;
    private readonly LandingHud _panel = new();
    private readonly NativeHud _native = new();
    private bool _nativeFailed;
    private HudModel _hudModel = new();
    private long _session = -1;
    private string _seenGrade = "";
    private float _seenGradeStars;
    private string _hud = "";
    private bool _flightValidated;
    private bool _degradedValidated;

    private bool Experimental
    {
        get
        {
#if LANDING_AUTO
            return _experimentalTests?.Value == true && _policy != null;
#else
            return false;
#endif
        }
    }

    private bool Authoritative
    {
        get
        {
#if LANDING_AUTO
            return _patched && _policy != null && (_flightValidated || Experimental);
#else
            return false;
#endif
        }
    }

    private static string TraceDirectory => Path.Combine(MelonEnvironment.UserDataDirectory, "MaydayLanding", Mode, "rec");

    public override void OnInitializeMelon()
    {
        var probeCategory = MelonPreferences.CreateCategory("MaydayLandingProbe", "Mayday Landing Probe");
        _probeEnabled = probeCategory.CreateEntry("Probe", ProbeDefault,
            "Record native gust/touchdown/hazard data and prior-vs-model feedback / 记录原生扰动与模型对比");
        _autoRecord = probeCategory.CreateEntry("AutoRecord", ProbeDefault,
            "Start and stop recordings automatically around each approach / 每次进近自动录制");
#if LANDING_AUTO
        var category = MelonPreferences.CreateCategory("MaydayLandingAuto", AutoName);
        _experimentalTests = category.CreateEntry("ExperimentalFlightTests", true,
            "Allow F7 to switch between advice and full auto with the experimental policy / 允许 F7 在建议与全自动之间切换");
        _autoModeEntry = category.CreateEntry("AutoMode", false,
            "Start in full auto (arms every approach) instead of advice / 启动时即为全自动");
        _autoMode = _autoModeEntry.Value;
#endif
        try
        {
            var assembly = typeof(LandingMod).Assembly;
            using var model = assembly.GetManifestResourceStream("MaydayLanding.Policy.json");
            if (model != null)
            {
                using var modelBytes = new MemoryStream();
                model.CopyTo(modelBytes);
                modelBytes.Position = 0;
                using var reader = new StreamReader(modelBytes);
                _policy = LandingPolicy.Load(reader.ReadToEnd());
                _policySha256 = Convert.ToHexString(SHA256.HashData(modelBytes.ToArray())).ToLowerInvariant();
                using var validation = assembly.GetManifestResourceStream("MaydayLanding.FlightValidated");
                using var degraded = assembly.GetManifestResourceStream("MaydayLanding.DegradedValidated");
                _flightValidated = validation != null;
                _degradedValidated = degraded != null;
            }
        }
        catch (Exception error)
        {
            _policy = null;
            _policySha256 = null;
            LoggerInstance.Warning("进近模型载入失败，已切换至手飞提示。 " + error.Message);
        }
    }

    public override void OnLateInitializeMelon()
    {
#if LANDING_AUTO
        if (_policy != null && (_flightValidated || Experimental))
        {
            try
            {
                _patchConflict = ControllerConflict.HasUnknownPatch(HarmonyInstance);
                LandingInput.Install(HarmonyInstance, CanDrive);
                _patched = true;
            }
            catch (Exception error)
            {
                LandingInput.Uninstall(HarmonyInstance);
                LoggerInstance.Warning("接管接口不可用，已切换至手飞提示。 " + error.Message);
            }
        }
#else
        CheckDuplicateAdvisor();
        if (_duplicateAdvisor) return;
#endif
        if (ProbeOn)
        {
#if LANDING_AUTO
            string? hookError = ProbeHooks.Install(HarmonyInstance);
#else
            string? hookError = "Advisor never patches the game; touchdown details come from the native grade";
#endif
            LoggerInstance.Msg(hookError == null ? "测试探针已启用（接地、危险事件、阵风、模型对比）。Shift+F6 标记。"
                : "测试探针已启用，但原生事件钩子不可用：" + hookError);
        }
        LoggerInstance.Msg($"{Name} 已加载。F6 记录，F7 {(Authoritative ? "接通／断开" : "显示／隐藏")}。记录目录 {TraceDirectory}");
    }

    public override void OnUpdate()
    {
#if !LANDING_AUTO
        CheckDuplicateAdvisor();
        if (_duplicateAdvisor) return;
#endif
        try
        {
            UpdateFlight();
            _updateError = false;
        }
        catch (Exception error)
        {
            ReleaseOutputs();
            _stopPending = true;
            ResetPolicy();
            _hud = "进近数据读取失败 · 请手动飞行";
            _hudModel = new HudModel { Mode = "故障", ModeTone = HudTone.Alert, Phase = "请手动飞行",
                Instruction = "进近数据读取失败", InstructionTone = HudTone.Alert, Controls = "F7 隐藏面板" };
            if (!_updateError)
            {
                LoggerInstance.Warning(_hud + "。 " + error.Message);
                if (_trace != null)
                {
                    var failure = new LandingFrame { SessionId = _session, Time = Time.unscaledTimeAsDouble };
                    WriteTrace(failure, _decision.Phase, "Runtime error. Fly manually.", false, null);
                }
            }
            _updateError = true;
        }
    }

    private void UpdateFlight()
    {
        double now = Time.unscaledTimeAsDouble;
        var keyboard = Application.isFocused && !UIState.TextInputFocused ? Keyboard.current : null;
        bool toggle = keyboard != null && keyboard[Key.F7].wasPressedThisFrame;
        bool goAround = keyboard != null && keyboard[Key.F5].wasPressedThisFrame;
        bool shift = keyboard != null && (keyboard[Key.LeftShift].isPressed || keyboard[Key.RightShift].isPressed);
        if (keyboard != null && keyboard[Key.F6].wasPressedThisFrame)
        {
            if (shift && ProbeOn) MarkTrace(now);
            else { _autoSuppressed = _trace != null; ToggleTrace(); }
        }
        if (toggle && !Authoritative) _hudEnabled = !_hudEnabled;
        if (toggle && Authoritative) SetAutoMode(!_autoMode && !_decision.Engaged);

        bool stop = _stopPending || (toggle && Authoritative && (_decision.Engaged || _decision.Armed));
        _stopPending = false;
        // In full auto, re-arm on every approach once airborne. The director acts on a rising edge,
        // so the request alternates frames until the arm is accepted.
        bool airborne = _frame != null && _frame.Valid && _frame["on_ground"] < 0.5f;
        bool wantArm = Authoritative && _autoMode && !_decision.Engaged && !_decision.Armed && !stop && airborne;
        _armPulse = wantArm && !_armPulse;
        bool arm = _armPulse;
        _game.HoldRunway = _decision.Engaged && _decision.Phase == LandingPhase.Rollout && !stop;
        var frame = _game.Read(ExternalController(now, arm));
#if LANDING_AUTO
        if (_patched)
        {
            frame.PilotOverride |= LandingInput.ConsumeManualRelease();
            stop |= LandingInput.ConsumeAuthorityLoss();
        }
#endif
        if (_session != frame.SessionId)
        {
            _session = frame.SessionId;
            _autoSuppressed = false;
            _seenGrade = _game.GradeKey;
            _seenGradeStars = _game.GradeStars;
            ReleaseOutputs();
            ResetPolicy();
        }

        if (_game.GradeKey.Length > 0 && (_game.GradeKey != _seenGrade || _game.GradeStars != _seenGradeStars))
            frame.TouchdownScore = _game.GradeStars;
        _seenGrade = _game.GradeKey;
        _seenGradeStars = _game.GradeStars;
        if (stop || arm) ResetPolicy();
        frame.RemoteResponseMonitored = Authoritative && !frame.Host && _game.OtherOccupied
            && frame.Valid && frame["observation_age"] < 0.3f;
        if (frame.RemoteResponseMonitored && _decision.Engaged && _response.Disagrees(frame))
            frame.OtherPilotOverride = true;

        bool wasEngaged = _decision.Engaged;
        LandingPhase priorPhase = _decision.Phase;
        _frame = frame;
        LandingRouteStage routeStage = frame.Valid ? RouteStage(NavTelemetry.Leg) : LandingRouteStage.Unknown;
        _decision = _director.Step(frame, new LandingIntent(arm, stop, goAround,
            Authoritative, _degradedValidated || Experimental, frame.ParkingBrakeSet, routeStage));
        if (!_decision.Engaged) ReleaseOutputs();
        if (wasEngaged && !_decision.Engaged) ResetPolicy();
        if (_decision.Phase == LandingPhase.GoAround)
        {
            frame["target_along"] = _decision.TargetAlong;
            frame["target_cross"] = _decision.TargetCross;
        }

        FillPreviousAction(frame);
        bool predictive = _policy != null && _decision.GuidanceActive && frame.Valid && frame.Seated && !frame.Crashed
            && frame["observation_age"] <= ObservationExpiry && !stop && !_decision.Released;
        if (predictive && (now - _proposalAt >= PolicyPeriod - 0.001 || _decision.Phase != priorPhase))
        {
            _proposal = _policy!.Predict(frame.Values, (int)_decision.Phase);
            _proposalAt = now;
            frame.PredictedNow = true;
            var stallSpeeds = ProbeOn ? _policy.PriorStallSpeeds : null;
            _priorAction = stallSpeeds != null ? LandingPrior.Predict(frame.Values, _decision.Phase, stallSpeeds) : null;
        }
        else if (!predictive && _proposal != null) ResetPolicy();
        frame.PolicyReset = _policyResetPending;
        _policyResetPending = false;

        float[]? sent = null;
#if LANDING_AUTO
        if (_decision.Engaged && _decision.Axes != LandingAxes.None && _proposal != null && _game.Plane != null)
        {
            var masked = Quantize(_proposal, _decision.Axes, _decision.HoldBrake);
            ManageReverse(_game.Plane, frame, now, _decision.UseReverse, masked);
            double expiry = Math.Min(_proposalAt + OutputExpiry, now + ActuationAgeLimit - frame["observation_age"]);
            if (LandingInput.Publish(_game.Plane, expiry, _decision.Axes, masked))
            {
                frame.AuthorizedAxes = (int)_decision.Axes;
                sent = masked;
                frame.ThrottleSent = SendThrottle(_game.Plane, now, masked[3], _decision.Axes);
                if ((_decision.Axes & LandingAxes.Throttle) != 0 && _lastThrottle >= 0)
                    masked[3] = _lastThrottle / 255f;
                if (_decision.SecureParkingBrake) SecureParkingBrake(_game.Plane, frame, now);
                if (frame.RemoteResponseMonitored && (frame.PredictedNow || _previousAxes != _decision.Axes))
                    _response.Add(now, masked);
            }
            else _stopPending = true;
        }
        else if (_patched) LandingInput.Release();
        if (_decision.DisconnectAp && Authoritative && _game.Plane != null && frame.NativeAp
            && now - _lastApDisconnect > 1.5 + 2 * frame["observation_age"])
        {
            // CmdPilotAction(3) -> ServerControl(5): disconnects the native autopilot only when it is engaged.
            _game.Plane.CmdPilotAction(3);
            _lastApDisconnect = now;
            LoggerInstance.Msg("已预位：断开原生自动驾驶以接管进近。");
        }
#endif
        string status = _stopPending ? "Output released. Fly manually." : _decision.Status;
        bool authorized = frame.AuthorizedAxes != 0;
        _previousIntent = sent;
        _previousAxes = (LandingAxes)frame.AuthorizedAxes;
        if (wasEngaged != authorized)
            LoggerInstance.Msg(authorized ? "已接通进近控制。" : "已交还操纵。 " + status);
        // Any hand-back other than a completed landing drops to advice, so releasing the stick
        // after a takeover never hands control straight back to the assist.
        if (wasEngaged && !_decision.Engaged && _autoMode && !status.StartsWith("Aircraft stopped.", StringComparison.Ordinal))
            SetAutoMode(false);
        AutoRecord(frame, now, authorized);
        ShowNativeHud();
        if (_trace != null) WriteTrace(frame, _decision.Released ? priorPhase : _decision.Phase, status, authorized, sent);
        BuildHud(frame, status, authorized);
    }

    private static LandingRouteStage RouteStage(NavLeg leg) => leg switch
    {
        NavLeg.ToEntry or NavLeg.ToFix or NavLeg.Final => LandingRouteStage.Approach,
        NavLeg.Rollout => LandingRouteStage.Rollout,
        NavLeg.None => LandingRouteStage.Unknown,
        _ => LandingRouteStage.Other
    };

    private bool ExternalController(double now, bool immediate = false)
    {
        if (_conflict.HasController(this)) return true;
#if LANDING_AUTO
        if (_patched && (immediate || now >= _checkPatchesAt))
        {
            _patchConflict = ControllerConflict.HasUnknownPatch(HarmonyInstance);
            _checkPatchesAt = now + 1d;
        }
        return _patchConflict;
#else
        return false;
#endif
    }

#if !LANDING_AUTO
    private void CheckDuplicateAdvisor()
    {
        bool duplicate = ControllerConflict.AutoInstalled(this);
        if (duplicate == _duplicateAdvisor) return;
        _duplicateAdvisor = duplicate;
        _hud = "";
        if (duplicate)
        {
            CloseTrace();
            LoggerInstance.Msg("已发现自动进近版，建议版共用热键与状态栏由自动进近版接管。");
        }
        ResetPolicy();
    }
#endif

#if LANDING_AUTO
    private bool CanDrive(PlaneFlight plane)
    {
        if (!Authoritative || !_decision.Engaged || _frame == null || !_frame.Valid || !_frame.Seated
            || _frame.PilotOverride || _frame.OtherPilotOverride || _frame.Crashed || _stopPending)
            return false;
        if (!Application.isFocused || UIState.BlocksGameplay || CockpitIntrusion.Seated
            || plane == null || !plane.HasRender || plane.Networkcrashed || (plane.NetworkapModes & 1) != 0
            || (!plane.isServer && !Il2CppMirror.NetworkClient.isConnected)
            || _game.Plane == null || _game.Plane.Pointer != plane.Pointer || ExternalController(Time.unscaledTimeAsDouble))
            return false;
        var activePlane = PlaneFlight.Instance;
        if (activePlane == null || activePlane.Pointer != plane.Pointer) return false;
        var pilot = PlayerNet.Local;
        if (pilot == null || pilot.netId != _game.PlayerId || (pilot.IsPiloting ? 1 : PlaneFlight.IsCopilot(pilot) ? 2 : 0) != _game.Seat)
            return false;
        if (plane.isServer && (_game.ReadOtherSeat(_frame, plane, _game.Seat) || _frame["other_stick"] > 0.12f))
            return false;
        return true;
    }

    private static float[] Quantize(float[] proposal, LandingAxes axes, bool holdBrake)
    {
        var action = new float[5];
        for (int i = 0; i < action.Length; i++)
        {
            if (((int)axes & (1 << i)) == 0) continue;
            action[i] = i < 3 ? FlightSnapshot.PackSigned(proposal[i]) / 127f : Mathf.Clamp01(proposal[i]);
        }
        action[3] = Mathf.RoundToInt(action[3] * 255f) / 255f;
        action[4] = action[4] >= 0.5f ? 1f : 0f;
        if (holdBrake)
        {
            action[3] = 0f;
            action[4] = 1f;
        }
        return action;
    }

    private float? SendThrottle(PlaneFlight plane, double now, float normalized, LandingAxes axes)
    {
        if ((axes & LandingAxes.Throttle) == 0 || now - _lastThrottleTime < 0.1) return null;
        int value = Mathf.RoundToInt(normalized * 255f);
        if (_lastThrottle >= 0 && Math.Abs(value - _lastThrottle) < 2 && value is not (0 or 255)) return null;
        if (_lastThrottle == value || plane.LocalHandOnThrottle) return null;
        plane.CmdSetThrottles((byte)value);
        _lastThrottle = value;
        _lastThrottleTime = now;
        return value / 255f;
    }

    /// <summary>Own reverse thrust only during an engaged rollout. Throttle is held at idle while the
    /// lever state is changing, and reverse is always switched back off before control is returned.</summary>
    private void ManageReverse(PlaneFlight plane, LandingFrame frame, double now, bool want, float[] masked)
    {
        if ((_decision.Axes & LandingAxes.Throttle) == 0) return;
        bool on = plane.NetworkreverseGear;
        if (!on) _reverseOwned &= now - _lastReverseTime < 1.2 + 2 * frame["observation_age"];
        bool settle = now - _lastReverseTime < 1.2 + 2 * frame["observation_age"];
        if (want && frame["on_ground"] >= 0.5f)
        {
            if (!on && !settle)
            {
                plane.CmdPilotAction(4);
                _lastReverseTime = now;
                _reverseOwned = true;
            }
            masked[3] = on && _reverseOwned ? ReverseThrottle : 0f;
        }
        else if (on && _reverseOwned)
        {
            masked[3] = 0f;
            if (!settle)
            {
                plane.CmdPilotAction(4);
                _lastReverseTime = now;
            }
        }
        frame.ReverseOn = on;
    }

    private void StowReverse()
    {
        var plane = _game.Plane;
        if (_reverseOwned && plane != null && plane.NetworkreverseGear) plane.CmdPilotAction(4);
        _reverseOwned = false;
        _lastReverseTime = double.NegativeInfinity;
    }

    private void SecureParkingBrake(PlaneFlight plane, LandingFrame frame, double now)
    {
        if (plane.NetworkparkingBrake || frame["ground_speed"] >= 0.1f
            || now - _lastParkTime < 1.2 + 2 * frame["observation_age"]) return;
        plane.CmdPilotAction(2);
        _lastParkTime = now;
        frame.ParkingBrakeRequested = true;
    }
#endif

    private void ReleaseOutputs()
    {
#if LANDING_AUTO
        if (_patched) LandingInput.Release();
        try { StowReverse(); } catch (Exception error) { LoggerInstance.Warning("反推收起失败，请手动关闭反推。 " + error.Message); }
        _lastThrottle = -1;
        _lastThrottleTime = double.NegativeInfinity;
        _lastParkTime = double.NegativeInfinity;
#endif
        _response.Reset();
        _previousIntent = null;
        _previousAxes = LandingAxes.None;
    }

    private void ResetPolicy()
    {
        _proposalAt = double.NegativeInfinity;
        _proposal = null;
        if (_policy != null)
        {
            _policy.Reset();
            _policyResetPending = true;
        }
    }

    private void FillPreviousAction(LandingFrame frame)
    {
        var raw = frame.RawInputs;
        bool owned = _previousIntent != null && !frame.PilotOverride && !frame.OtherPilotOverride
            && !frame.ExternalController && !frame.NativeAp;
        frame["last_pitch"] = Previous(LandingAxes.Pitch, 0, raw?[0] ?? 0f);
        frame["last_roll"] = Previous(LandingAxes.Roll, 1, raw?[1] ?? 0f);
        frame["last_yaw"] = Previous(LandingAxes.Yaw, 2, raw?[2] ?? 0f);
        frame["last_throttle"] = Previous(LandingAxes.Throttle, 3, frame["throttle"]);
        frame["last_brake"] = Previous(LandingAxes.Brake, 4, raw?[4] ?? 0f);

        float Previous(LandingAxes axis, int index, float fallback) =>
            owned && (_previousAxes & axis) != 0 ? _previousIntent![index] : fallback;
    }

    private bool ProbeOn => _probeEnabled?.Value == true;

    private void AutoRecord(LandingFrame frame, double now, bool authorized)
    {
        if (!ProbeOn || _autoRecord?.Value != true) return;
        bool context = frame.Valid && (_decision.GuidanceActive || authorized);
        if (_trace == null)
        {
            if (context && !_autoSuppressed)
            {
                ToggleTrace();
                _autoTrace = _trace != null;
                _outsideSince = double.NaN;
            }
            return;
        }
        if (!_autoTrace) return;
        if (context) _outsideSince = double.NaN;
        else if (double.IsNaN(_outsideSince)) _outsideSince = now;
        else if (now - _outsideSince > AutoRecordLinger) CloseTrace();
    }

    private void MarkTrace(double now)
    {
        if (_trace == null) ToggleTrace();
        _probe?.Event("mark", now);
        LoggerInstance.Msg("已在记录中标记当前时刻。");
    }

    private ProbeNative? ReadNative()
    {
        var plane = _game.Plane;
        if (plane == null || !plane.isServer) return null;
        return new ProbeNative(plane.pendingGust, plane.gustDir, plane.gustT, plane.hazardTurb, plane.joltT, plane.joltAccel);
    }

    private void ToggleTrace()
    {
        if (_trace != null) { CloseTrace(); return; }
        try
        {
            _trace = LandingTrace.Open(TraceDirectory, _policySha256, ProbeOn ? LandingProbe.Schema : null);
            _probe = ProbeOn ? new LandingProbe() : null;
            _autoTrace = false;
            _traceError = null;
            LoggerInstance.Msg("已开始记录进近。 " + TraceDirectory);
        }
        catch (Exception error)
        {
            _traceError = "记录无法启动";
            LoggerInstance.Warning(_traceError + "。 " + error.Message);
        }
    }

    private void WriteTrace(LandingFrame frame, LandingPhase phase, string status, bool authorized, float[]? sent)
    {
        try
        {
            object? probe = null;
            if (_probe != null)
            {
#if LANDING_AUTO
                ProbeHooks.Drain(_probe, frame.Time);
#endif
                ProbeNative? native = null;
                try { native = ReadNative(); } catch { }
                probe = _probe.Observe(frame, native, frame.PredictedNow ? _proposal : null,
                    frame.PredictedNow ? _priorAction : null, authorized, status);
            }
            _trace?.Write(frame, (int)phase, status, authorized, _proposal, sent, probe);
        }
        catch (Exception error)
        {
            _traceError = "记录写入失败";
            CloseTrace();
            LoggerInstance.Warning(_traceError + "。 " + error.Message);
        }
    }

    private void CloseTrace()
    {
        var trace = _trace;
        var probe = _probe;
        _trace = null;
        _probe = null;
        _autoTrace = false;
        try
        {
            if (trace != null && probe != null) trace.WriteSummary(probe.Summary(_policySha256, Mode));
        }
        catch (Exception error) { LoggerInstance.Warning("探针摘要写入失败。 " + error.Message); }
        try { trace?.Dispose(); }
        catch (Exception error) { LoggerInstance.Warning("关闭进近记录时出错。 " + error.Message); }
    }

    private void BuildHud(LandingFrame frame, string status, bool authorized)
    {
        bool guidance = _decision.GuidanceActive;
        var model = new HudModel();
        if (authorized) { model.Mode = Authoritative && _autoMode ? "全自动 · 接管" : "接管中"; model.ModeTone = HudTone.Engaged; }
        else if (_decision.Armed) { model.Mode = "全自动 · 预位"; model.ModeTone = HudTone.Armed; }
        else if (Authoritative && _autoMode) { model.Mode = "全自动"; model.ModeTone = HudTone.Armed; }
        else { model.Mode = "建议"; model.ModeTone = guidance ? HudTone.Guide : HudTone.Monitor; }
        model.Phase = !frame.Valid ? "等待飞行数据" : !guidance ? "进近待命" : _decision.Phase switch
        {
            LandingPhase.Capture => "对准进近",
            LandingPhase.Final => "最后进近",
            LandingPhase.Flare => "拉平",
            LandingPhase.Rollout => "着陆滑跑",
            LandingPhase.GoAround => "复飞",
            _ => "进近"
        };

        string instruction = TranslateStatus(status);
        if (guidance && !authorized && !Authoritative && _policy == null) instruction += "　无进近模型";
        bool configWarning = false;
        if (frame.Valid && _decision.LandingConfigurationGuidance && frame["on_ground"] < 0.5f && frame["agl"] < 140f)
        {
            if (frame["gear"] < 0.9f) { instruction += "　检查起落架"; configWarning = true; }
            if (frame["flap"] < 1.9f) { instruction += "　检查襟翼"; configWarning = true; }
        }
        // Red only for things that need the pilot now; "no approach data" on the ground is not an alert.
        bool alert = _decision.Released || configWarning
            || status.Contains(LandingDirector.JammedGearCue, StringComparison.Ordinal)
            || ((guidance || authorized) && status.Contains("Fly manually", StringComparison.Ordinal));
        bool seated = _game.Seat != 0;
        if (!seated)
        {
            model.Phase = "未在驾驶座";
            instruction = Authoritative ? "坐到机长或副驾驶座即可使用建议与全自动" : "坐到机长或副驾驶座即可看到进近建议";
            alert = false;
        }
        else if (!frame.Valid && !authorized)
            instruction = _decision.Armed || (Authoritative && _autoMode)
                ? "全自动待命：选定目的地跑道并进入进近后自动接管"
                : "等待进近数据：选定目的地跑道并进入进近后显示建议";
        model.Instruction = instruction;
        model.InstructionTone = alert ? HudTone.Alert : _decision.Armed ? HudTone.Armed : authorized ? HudTone.Engaged : HudTone.Monitor;
        if (_decision.Released) { model.Mode = "已交还 · 建议"; model.ModeTone = HudTone.Alert; }

        if (frame.Valid)
        {
            if (guidance || authorized || _decision.Armed)
            {
                string rings = _game.MissedGates > 0 ? $"{_game.PassedGates}/5 漏{_game.MissedGates}" : $"{_game.PassedGates}/5";
                model.Metrics.Add(("环", rings, _game.MissedGates > 0));
            }
            else model.Metrics.Add(("距入口", $"{-frame["along"] / 1000f:0.0} km", false));
            model.Metrics.Add(("离地", $"{frame["agl"]:0} m", false));
            model.Metrics.Add(("侧偏", $"{frame["cross"]:+0;-0;0} m", MathF.Abs(frame["cross"]) > 60f && frame["agl"] < 140f));
            model.Metrics.Add(("下沉", $"{frame["sink"]:+0.0;-0.0;0.0}", frame["sink"] > 3f && frame["agl"] < 30f));
            if (frame.Seated && frame["other_known"] < 0.5f) model.Detail = "另一座输入不可见";
        }
        if (authorized)
        {
            var axes = (LandingAxes)frame.AuthorizedAxes;
            var owned = new List<string>(5);
            if ((axes & LandingAxes.Pitch) != 0) owned.Add("俯仰");
            if ((axes & LandingAxes.Roll) != 0) owned.Add("横滚");
            if ((axes & LandingAxes.Yaw) != 0) owned.Add("偏航");
            if ((axes & LandingAxes.Throttle) != 0) owned.Add(frame.ReverseOn ? "油门（反推）" : "油门");
            if ((axes & LandingAxes.Brake) != 0) owned.Add("刹车");
            model.Detail = (model.Detail.Length > 0 ? model.Detail + "　" : "") + "接管：" + string.Join(" ", owned);
        }
        model.Controls = Authoritative
            ? (_decision.Engaged ? "F7 断开并回到建议　F5 复飞" : _autoMode ? "F7 切到建议" : "F7 切到全自动")
            : "F7 隐藏面板";
        if (_trace != null)
        {
            model.Recording = "● " + (_autoTrace ? "自动记录中" : "记录中") + "　F6 停止" + (ProbeOn ? "　Shift+F6 标记" : "");
            model.RecordingActive = true;
        }
        else model.Recording = _traceError ?? "F6 记录";
        _hudModel = model;
        _hud = model.Phase;
    }

    private static string TranslateStatus(string status)
    {
        const string jammedGear = "起落架卡滞，尚未锁定。着陆前请手摇放下起落架，确认锁定。";
        if (status.StartsWith(LandingDirector.JammedGearCue + " Fly manually.", StringComparison.Ordinal))
            return TranslatePrefix(LandingDirector.JammedGearCue + " Fly manually.", "请手动飞行 · " + jammedGear);
        if (status.StartsWith("Arming cancelled.", StringComparison.Ordinal)) return "已取消预位 · 请手动飞行";
        if (status.StartsWith("Armed. Disconnecting native autopilot.", StringComparison.Ordinal)) return "进入接管范围 · 正在断开原生自动驾驶";
        if (status.StartsWith("Armed. Release the controls to engage.", StringComparison.Ordinal)) return "进入接管范围 · 松开操纵杆即可自动接管";
        if (status.StartsWith("Armed.", StringComparison.Ordinal)) return "等待进入接管范围：" + TranslateGaps(status["Armed.".Length..]);
        if (status.StartsWith("Pilot has control.", StringComparison.Ordinal)) return "驾驶员已接管";
        if (status.StartsWith("Another controller has control.", StringComparison.Ordinal)) return TranslatePrefix("Another controller has control.", "另一个飞控正在操纵飞机");
        if (status.StartsWith("Native autopilot has control.", StringComparison.Ordinal)) return TranslatePrefix("Native autopilot has control.", "原版自动驾驶正在操纵飞机");
        if (status.StartsWith("Assist off.", StringComparison.Ordinal)) return "进近控制已断开 · 请手动飞行";
        if (status.StartsWith("Output released.", StringComparison.Ordinal)) return "已释放进近控制 · 请手动飞行";
        if (status.StartsWith("Flight data", StringComparison.Ordinal)) return "飞行数据变化、不可用或延迟 · 请手动飞行";
        if (status.StartsWith("Copilot controls unverified or in use. Fly manually.", StringComparison.Ordinal))
            return TranslatePrefix("Copilot controls unverified or in use. Fly manually.", "另一驾驶位输入未确认或正在操纵 · 请手动飞行");
        if (status.StartsWith("Aircraft crashed", StringComparison.Ordinal)) return "飞机已坠毁 · 控制已释放";
        if (status.StartsWith("Capturing approach.", StringComparison.Ordinal)) return TranslatePrefix("Capturing approach.", "正在对准进近航线");
        if (status.StartsWith("Final approach.", StringComparison.Ordinal)) return TranslatePrefix("Final approach.", "沿下滑道进近");
        if (status.StartsWith("Flare.", StringComparison.Ordinal)) return TranslatePrefix("Flare.", "拉平减小接地下沉率");
        if (status.StartsWith("Manual guidance.", StringComparison.Ordinal)) return TranslatePrefix("Manual guidance.", "手飞提示");
        if (status.StartsWith("Runway rollout. Reverse thrust.", StringComparison.Ordinal)) return "保持跑道中线 · 反推减速";
        if (status.StartsWith("Runway rollout.", StringComparison.Ordinal)) return "保持跑道中线并减速";
        if (status.StartsWith("Holding brake.", StringComparison.Ordinal)) return "保持刹车";
        if (status.StartsWith("Securing parking brake.", StringComparison.Ordinal)) return "正在设置停机刹车";
        if (status.StartsWith("Aircraft stopped.", StringComparison.Ordinal)) return "飞机停稳，停机刹车已确认";
        if (status.Contains("Fly manually.", StringComparison.Ordinal)) return "请手动飞行　" + status;
        return status.Replace(LandingDirector.JammedGearCue, jammedGear, StringComparison.Ordinal);

        string TranslatePrefix(string prefix, string text)
        {
            string suffix = status[prefix.Length..].Trim().Replace(LandingDirector.JammedGearCue, jammedGear, StringComparison.Ordinal);
            return suffix.Length == 0 ? text : text + "　" + suffix;
        }
    }

    private static string TranslateGaps(string text)
    {
        var parts = new List<string>();
        foreach (string token in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            string[] kv = token.Split(':', 2);
            string[] vl = kv.Length > 1 ? kv[1].Split('/', 2) : new[] { "" };
            string value = vl[0], limit = vl.Length > 1 ? vl[1] : "";
            parts.Add(kv[0] switch
            {
                "distance" => $"距入口 {value} km（≤ {limit} km 时接管）",
                "too-close" => "离跑道太近",
                "cross" => $"横向偏差 {value} m（需 ≤ {limit} m）",
                "glide" => $"偏离下滑道 {value} m（需 ±{limit} m）",
                "low" => "高度过低",
                "speed" => $"空速 {value} m/s（需 {limit}）",
                "track" => $"航迹夹角 {value}°（需 ≤ {limit}°）",
                "bank" => $"坡度 {value}°（需 ≤ {limit}°）",
                "stall" => "失速告警",
                "vertical-speed" => "升降率过大",
                "airborne" => "需在空中",
                _ => token
            });
        }
        return parts.Count == 0 ? text.Trim() : string.Join("，", parts);
    }

    public override void OnGUI()
    {
#if !LANDING_AUTO
        if (_duplicateAdvisor) return;
#endif
        if (!HudVisible || _native.Active) return;
        _panel.Draw(_hudModel);
    }

    private bool HudVisible => _hudEnabled && !string.IsNullOrEmpty(_hud);

    private void SetAutoMode(bool on)
    {
        if (_autoMode == on) return;
        _autoMode = on;
        LoggerInstance.Msg(on ? "已切换到全自动：进入接管范围后自动接管。" : "已切换到建议模式。");
        if (_autoModeEntry != null) { _autoModeEntry.Value = on; MelonPreferences.Save(); }
    }

    /// <summary>Prefer the game-native panel; any failure falls back to the IMGUI panel for the session.</summary>
    private void ShowNativeHud()
    {
        if (_nativeFailed) return;
        try
        {
            if (_native.Ensure()) _native.Show(_hudModel, HudVisible);
        }
        catch (Exception error)
        {
            _nativeFailed = true;
            try { _native.Destroy(); } catch { }
            LoggerInstance.Warning("原生面板不可用，改用简易面板。 " + error.Message);
        }
    }

    public override void OnSceneWasUnloaded(int buildIndex, string sceneName)
    {
        ReleaseOutputs();
        _stopPending = true;
        ResetPolicy();
    }

    public override void OnApplicationQuit()
    {
        ReleaseOutputs();
        CloseTrace();
    }

    public override void OnDeinitializeMelon()
    {
        ReleaseOutputs();
        CloseTrace();
#if LANDING_AUTO
        if (_patched) LandingInput.Uninstall(HarmonyInstance);
#endif
        _panel.Dispose();
        try { _native.Destroy(); } catch { }
    }

    private sealed class MotionResponse
    {
        private readonly Queue<(double At, float[] Axes)> _commands = new();
        private (double At, float[] Axes)? _candidate;
        private double? _lastSource;
        private double _lastObservation;
        private double _divergence;

        public void Add(double at, float[] axes)
        {
            _commands.Enqueue((at, axes));
            while (_commands.Count > 120) _commands.Dequeue();
        }

        public bool Disagrees(LandingFrame frame)
        {
            if (!frame.MotionSourceTime.HasValue || frame.MotionSourceTime == _lastSource) return false;
            if (_lastSource.HasValue && frame.MotionSourceTime.Value < _lastSource.Value) { Reset(); return false; }
            _lastSource = frame.MotionSourceTime;
            double dt = Math.Clamp(frame.Time - _lastObservation, 0d, 0.15d);
            _lastObservation = frame.Time;
            double commandedAt = frame.Time - frame["observation_age"] - 0.18;
            while (_commands.TryPeek(out var command) && command.At <= commandedAt)
                _candidate = _commands.Dequeue();
            if (_candidate is not { } expected || commandedAt - expected.At > 0.25) { _divergence = 0d; return false; }
            bool disagrees = frame["on_ground"] > 0.5f
                ? frame["ground_speed"] >= 12f && Weak(expected.Axes[2], frame["yaw_rate"], 0.8f)
                : Weak(expected.Axes[0], frame["pitch_rate"], 0.8f) || Weak(expected.Axes[1], frame["roll_rate"], 2f);
            _divergence = disagrees ? _divergence + dt : 0d;
            return _divergence >= 0.9;
        }

        private static bool Weak(float command, float rate, float minimum) =>
            MathF.Abs(command) >= 0.5f && MathF.Sign(command) * rate < minimum;

        public void Reset()
        {
            _commands.Clear();
            _candidate = null;
            _lastSource = null;
            _lastObservation = _divergence = 0d;
        }
    }
}
