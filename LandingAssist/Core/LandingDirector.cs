using System.Collections.Generic;
using System;

namespace MaydayLanding;

public enum LandingPhase { Capture = 0, Final = 1, Flare = 2, Rollout = 3, GoAround = 4 }
public enum LandingRouteStage { Unknown, Other, Approach, Rollout }

[Flags]
public enum LandingAxes { None = 0, Pitch = 1, Roll = 2, Yaw = 4, Throttle = 8, Brake = 16 }

public readonly record struct LandingIntent(
    bool Engage = false, bool Stop = false, bool GoAround = false, bool PolicyValidated = false,
    bool SupportsDegraded = false, bool ParkingBrakeSet = false, LandingRouteStage RouteStage = LandingRouteStage.Unknown);

public readonly record struct LandingDecision(
    LandingPhase Phase, LandingAxes Axes, string Status, bool Engaged, bool Released,
    int Reentries, float TargetAlong, float TargetCross)
{
    public bool HoldBrake { get; init; }
    public bool SecureParkingBrake { get; init; }
    /// <summary>Engaged rollout above ReverseOffSpeed with both engines normal: request reverse thrust.</summary>
    public bool UseReverse { get; init; }
    /// <summary>Armed and inside the capture window while the native autopilot flies: disconnect it.</summary>
    public bool DisconnectAp { get; init; }
    /// <summary>Waiting to engage automatically once the capture window is met.</summary>
    public bool Armed { get; init; }
    public bool GuidanceActive { get; init; }
    public bool LandingConfigurationGuidance => GuidanceActive && Phase is LandingPhase.Capture or LandingPhase.Final or LandingPhase.Flare;
}

/// <summary>Controls authorization and selects targets. The accepted policy supplies all flight commands.</summary>
public sealed class LandingDirector
{
    internal const string JammedGearCue = "Landing gear jammed before lock. Use the hand crank before landing.";
    private const float MaxActuationAge = 0.3f;
    private const float MaxObservationAge = 0.6f;
    private const float MinCaptureAgl = 65f;
    private const float FinalAgl = 140f;
    private const float FlareAgl = 10f;
    private const float MinGoAroundAgl = 25f;
    private const float GearLocked = 0.9f;
    private const float StopSpeed = 0.1f;
    private const float BrakeHoldSpeed = 0.75f;
    internal const float ReverseOffSpeed = 10f;
    private const double StopSeconds = 3;
    private const float MaxCircuitSeconds = 300f;
    private const int GateMissBits = 0b10_10_10_10_10;
    private const LandingAxes FlightAxes = LandingAxes.Pitch | LandingAxes.Roll | LandingAxes.Yaw | LandingAxes.Throttle;
    private static readonly string[] Required =
    {
        "along", "cross", "agl", "speed", "ground_speed", "sink", "pitch", "roll", "track_sin", "track_cos",
        "heading_sin", "heading_cos", "runway_length", "runway_half_width", "next_gate_index", "next_gate_distance",
        "gear", "flap", "stall_warning", "on_ground", "other_known", "other_stick", "observation_age",
        "gate_height_error", "turbulence", "ice", "wind_cross", "throttle", "last_throttle"
    };

    private enum CircuitLeg { Climb, Crosswind, Downwind, Return, Reorient }
    private enum Power { Unknown, Incapable, SingleEngine, Derated, Normal }

    private bool _engaged;
    private bool _armed;
    private bool _engageDown;
    private bool _goAroundDown;
    private bool _hadFrame;
    private bool _lastUsable;
    private bool _previousHost;
    private bool _followUp;
    private long _sessionId;
    private double _previousTime;
    private double? _previousMotionTime;
    private double _goAroundAt;
    private double _stoppedAt = double.NaN;
    private float _previousAlong;
    private float _previousCross;
    private float _previousAgl;
    private float _previousRunwayLength;
    private float _previousRunwayHalfWidth;
    private float _previousSpeed;
    private float _previousSink;
    private float _speedTrend;
    private float _sinkTrend;
    private double _trendSeconds;
    private double _highPowerSeconds;
    private int _reentries;
    private LandingPhase _phase;
    private CircuitLeg _circuit;

    public LandingDecision Step(LandingFrame frame, LandingIntent intent = default)
    {
        bool usable = Usable(frame);
        bool advisory = usable && !frame.Crashed && InLandingContext(frame, intent.RouteStage);
        var decision = Decide(frame, intent, usable, advisory);
        return decision with { GuidanceActive = decision.Engaged || (advisory && !GearJammedInFlight(frame)),
            Armed = _armed && !decision.Engaged };
    }

    private LandingDecision Decide(LandingFrame frame, LandingIntent intent, bool usable, bool advisory)
    {
        bool engage = intent.Engage && !_engageDown;
        bool goAround = intent.GoAround && !_goAroundDown;
        _engageDown = intent.Engage;
        _goAroundDown = intent.GoAround;

        bool sessionChanged = _hadFrame && (frame.SessionId != _sessionId || frame.Host != _previousHost);
        bool motionRewound = frame.MotionSourceTime is double source && _previousMotionTime is double priorSource &&
            source < priorSource - 0.05;
        bool discontinuity = _hadFrame && (sessionChanged || !double.IsFinite(frame.Time) ||
            frame.Time < _previousTime || frame.Time - _previousTime > 1.0 || motionRewound ||
            (usable && _lastUsable && Jumped(frame, frame.Time - _previousTime)));
        Remember(frame, usable, discontinuity);

        if (_armed && (intent.Stop || frame.Crashed || sessionChanged || frame["on_ground"] >= 0.5f))
            _armed = false;
        if (intent.Stop) return Release(frame, "Assist off. Fly manually.", usable);
        if (engage && _armed && !_engaged)
        {
            _armed = false;
            return Observe(frame, "Arming cancelled. Fly manually.", usable);
        }
        if ((_engaged || engage) && (frame.PilotOverride || frame.OtherPilotOverride))
            return Release(frame, "Pilot has control.", usable);
        if (frame.Crashed) return Release(frame, "Aircraft crashed.", usable);
        if (discontinuity && (_engaged || engage))
            return Release(frame, "Flight data changed. Fly manually.", usable);
        if (!frame.Seated)
            return ReadOnly(frame, "Observer mode.", usable, advisory);
        if (frame.ExternalController) return ReadOnly(frame, "Another controller has control.", usable, advisory);
        if (frame.NativeAp && _armed && usable && intent.PolicyValidated && CaptureWindow(frame) && !frame.PilotOverride)
            return Observe(frame, "Armed. Disconnecting native autopilot.", true) with { DisconnectAp = true };
        if (frame.NativeAp) return ReadOnly(frame, "Native autopilot has control.", usable, advisory);
        if (!usable) return Release(frame, "Flight data unavailable or delayed. Fly manually.", false);
        if (frame.SessionId == 0)
            return Release(frame, Advice(frame, AvailablePower(frame), "Control identity unavailable.", advisory), true);
        if (!EngageableCrew(frame))
            return Release(frame, Advice(frame, AvailablePower(frame), "Copilot controls unverified or in use. Fly manually.", advisory), true);

        Power power = AvailablePower(frame);
        string powerHint = PowerHint(frame, power);
        bool powerAllowsControl = power == Power.Normal ||
            (intent.SupportsDegraded && power is Power.SingleEngine or Power.Derated && SustainableApproach(frame));
        if (!intent.PolicyValidated)
            return Release(frame, Advice(frame, power, "Manual guidance. No accepted policy.", advisory), true);
        if (frame["observation_age"] > MaxActuationAge)
            return Release(frame, Advice(frame, power, "Control latency exceeded. Fly manually.", advisory), true);

        bool onGround = frame["on_ground"] >= 0.5f;
        if (_engaged && onGround)
            return RollOut(frame, intent.ParkingBrakeSet);
        if (_engaged && _phase != LandingPhase.GoAround && !InApproachArea(frame))
            return Release(frame, "Outside approach area. Fly manually.", true);

        if (!_engaged)
        {
            if (!engage && !_armed) return Observe(frame, Advice(frame, power, "Manual guidance.", advisory), true);
            if (!powerAllowsControl)
            {
                _armed |= engage;
                return Observe(frame, "Armed. " + DescribeUnavailablePower(power, powerHint), true);
            }
            if (!advisory || !CaptureWindow(frame) || (frame.GateMissed && frame["next_gate_index"] >= 5f))
            {
                // F7 outside the window arms instead of refusing; engagement follows automatically.
                _armed |= engage;
                return Observe(frame, "Armed. " + string.Join(" ", CaptureGaps(frame)), true);
            }
            if (_armed && frame.PilotOverride)
                return Observe(frame, "Armed. Release the controls to engage.", true);
            _armed = false;
            _engaged = true;
            _phase = ApproachPhase(frame);
            _reentries = 0;
            _followUp = false;
            _stoppedAt = double.NaN;
        }

        if (!powerAllowsControl)
            return Release(frame, DescribeUnavailablePower(power, powerHint), true);

        if (_phase == LandingPhase.GoAround)
            return ContinueCircuit(frame, power, powerHint);
        if (_phase == LandingPhase.Rollout)
        {
            if (!OnRunway(frame) || frame["agl"] > FlareAgl + 6f || Unstable(frame))
                return Release(frame, "Aircraft airborne outside the touchdown corridor. Fly manually.", true);
            _phase = LandingPhase.Flare;
            _stoppedAt = double.NaN;
            return Command("Touchdown bounce. Flare." + powerHint);
        }

        bool gateMissed = (frame.GateMask & GateMissBits) != 0 || (!_followUp && frame.GateMissed);

        bool unstable = Unstable(frame);
        if (goAround || gateMissed || unstable)
        {
            string reason = goAround ? "Go-around requested." : gateMissed ? "Approach ring missed." : "Unstable approach.";
            return BeginCircuit(frame, power, reason, powerHint);
        }

        LandingPhase observed = ApproachPhase(frame);
        if (observed > _phase) _phase = observed;
        if (_phase == LandingPhase.Flare && (frame["agl"] > FlareAgl + 6f || frame["along"] > frame["runway_length"] * 0.7f))
            return Release(frame, "Flare outside the landing corridor. Fly manually.", true);

        string stage = _phase switch
        {
            LandingPhase.Capture => "Capturing approach.",
            LandingPhase.Final => "Final approach.",
            _ => "Flare."
        };
        if (GearJammedInFlight(frame))
            stage += " " + JammedGearCue;
        else if (frame["gear"] < GearLocked && frame["agl"] <= FinalAgl)
            stage += " Check landing gear.";
        return Command(stage + powerHint);
    }

    private LandingDecision BeginCircuit(LandingFrame frame, Power power, string reason, string powerHint)
    {
        if (_followUp)
            return Release(frame, reason + " Retry used. Fly manually.", true);
        if (!GoAroundWindow(frame))
            return Release(frame, reason + " Automatic go-around outside its envelope. Fly manually.", true);
        if (!ClimbAvailable(frame, power))
            return Release(frame, reason + " Automatic climb unproven. Fly manually." + powerHint, true);

        _phase = LandingPhase.GoAround;
        _circuit = CircuitLeg.Climb;
        _goAroundAt = frame.Time;
        return CircuitCommand(reason + " Go-around climb." + powerHint);
    }

    private LandingDecision ContinueCircuit(LandingFrame frame, Power power, string powerHint)
    {
        if (!ClimbAvailable(frame, power))
            return Release(frame, "Go-around power margin lost. Fly manually." + powerHint, true);
        if (frame.Time - _goAroundAt > MaxCircuitSeconds || frame["agl"] < 12f || frame["stall_warning"] >= 0.5f ||
            MathF.Abs(frame["roll"]) > 40f)
            return Release(frame, "Go-around outside its envelope. Fly manually.", true);

        float along = frame["along"];
        float cross = frame["cross"];
        float agl = frame["agl"];
        switch (_circuit)
        {
            case CircuitLeg.Climb when along >= 1450f && agl >= 110f && frame["sink"] <= 0f:
                _circuit = CircuitLeg.Crosswind;
                break;
            case CircuitLeg.Crosswind when cross >= 1200f:
                _circuit = CircuitLeg.Downwind;
                break;
            case CircuitLeg.Downwind when along <= -4300f:
                _circuit = CircuitLeg.Return;
                break;
            case CircuitLeg.Return when along <= -4400f && MathF.Abs(cross) <= 460f:
                _circuit = CircuitLeg.Reorient;
                break;
            case CircuitLeg.Reorient when along <= -3850f && MathF.Abs(cross) <= 220f && CaptureWindow(frame):
                // Require a new ring set before attempting an approach after a miss.
                if ((frame.GateMask & GateMissBits) != 0 || frame["next_gate_index"] != 0)
                    return Release(frame, "Approach rings did not reset. Fly manually.", true);
                _followUp = true;
                _reentries++;
                _phase = LandingPhase.Capture;
                return Command("Retrying approach." + powerHint);
        }

        return CircuitCommand(_circuit switch
        {
            CircuitLeg.Climb => "Go-around climb.",
            CircuitLeg.Crosswind => "Go-around crosswind.",
            CircuitLeg.Downwind => "Go-around downwind.",
            _ => "Rejoining approach."
        } + powerHint);
    }

    private LandingDecision CircuitCommand(string status)
    {
        // Targets in the policy's physical-threshold runway coordinates.
        var (along, cross) = _circuit switch
        {
            CircuitLeg.Climb => (1750f, 0f),
            CircuitLeg.Crosswind => (1750f, 1500f),
            CircuitLeg.Downwind => (-4650f, 1500f),
            CircuitLeg.Return => (-4850f, 0f),
            _ => (-3500f, 0f)
        };
        return Command(status, along, cross);
    }

    private LandingDecision RollOut(LandingFrame frame, bool parkingBrakeSet)
    {
        if ((_phase != LandingPhase.Flare && _phase != LandingPhase.Rollout) || !OnRunway(frame))
            return Release(frame, "Touchdown outside the rollout corridor. Fly manually.", true);
        _phase = LandingPhase.Rollout;
        float speed = frame["ground_speed"];
        if (speed >= StopSpeed) _stoppedAt = double.NaN;
        else if (!double.IsFinite(_stoppedAt)) _stoppedAt = frame.Time;
        bool stopped = double.IsFinite(_stoppedAt) && frame.Time - _stoppedAt >= StopSeconds;
        if (stopped && parkingBrakeSet)
            return Release(frame, "Aircraft stopped. Assist complete.", true);
        if (speed <= BrakeHoldSpeed)
            return new LandingDecision(_phase, LandingAxes.Throttle | LandingAxes.Brake, stopped ? "Securing parking brake." : "Holding brake.",
                true, false, _reentries, 0f, 0f) { HoldBrake = true, SecureParkingBrake = stopped };
        bool reverse = speed > ReverseOffSpeed && AvailablePower(frame) == Power.Normal;
        return Command(reverse ? "Runway rollout. Reverse thrust." : "Runway rollout.") with { UseReverse = reverse };
    }

    private static bool OnRunway(LandingFrame frame) => frame["along"] >= 0f &&
        frame["along"] <= frame["runway_length"] && MathF.Abs(frame["cross"]) < frame["runway_half_width"];

    private static bool GearJammedInFlight(LandingFrame frame) =>
        frame.GearJammed && frame["on_ground"] < 0.5f && frame["gear"] < GearLocked;

    private LandingDecision Command(string text, float targetAlong = 0f, float targetCross = 0f) =>
        new(_phase, _phase == LandingPhase.Rollout ? FlightAxes | LandingAxes.Brake : FlightAxes,
            text, true, false, _reentries, targetAlong, targetCross);

    private LandingDecision Release(LandingFrame frame, string text, bool usable)
    {
        bool released = _engaged;
        _engaged = false;
        return Observe(frame, text, usable, released);
    }

    private LandingDecision ReadOnly(LandingFrame frame, string headline, bool usable, bool advisory) =>
        Release(frame, usable ? Advice(frame, AvailablePower(frame), headline, advisory) : headline + " Flight data unavailable.", usable);

    private LandingDecision Observe(LandingFrame frame, string text, bool usable, bool released = false) =>
        new(usable ? ApproachPhase(frame) : LandingPhase.Capture, LandingAxes.None, text, false, released,
            _reentries, 0f, 0f);

    private static LandingPhase ApproachPhase(LandingFrame frame)
    {
        if (frame["on_ground"] >= 0.5f) return LandingPhase.Rollout;
        if (frame["agl"] <= FlareAgl) return LandingPhase.Flare;
        float glideError = frame["agl"] + (frame["along"] - 250f) * 0.052408f;
        bool aligned = MathF.Abs(frame["cross"]) < 130f && frame["track_cos"] >= 0.913545f && MathF.Abs(glideError) < 35f;
        return aligned || frame["next_gate_index"] > 0 ? LandingPhase.Final : LandingPhase.Capture;
    }

    // Capture window from the twin evaluation of odd-speed / odd-angle starts (wide-1200k, plan §9):
    // before ring 1 (at 3750 m) a start needs >= 1 km to the ring and |cross| <= 0.3 x that distance;
    // that kept ring misses at 0.5% and success at 96.5% (single-engine included). Closer in, the
    // original narrow window applies.
    internal const float CaptureMaxDistance = 8500f, CaptureMaxCross = 600f, CaptureMaxGlideError = 150f;
    internal const float CaptureMinSpeed = 35f, CaptureMaxSpeed = 70f, CaptureMinTrackCos = 0.707f, CaptureMaxRoll = 25f;
    private const float FirstRingAlong = -3750f, WideWindowAlong = -4750f, CrossPerRingDistance = 0.3f;

    private static float GlideError(LandingFrame frame) => frame["agl"] + (frame["along"] - 250f) * 0.052408f;

    private static bool CaptureWindow(LandingFrame frame) => CaptureGaps(frame).Count == 0;

    /// <summary>Unmet capture conditions as short English tokens the HUD translates ("name:value/limit").</summary>
    internal static List<string> CaptureGaps(LandingFrame frame)
    {
        var gaps = new List<string>();
        if (frame["on_ground"] >= 0.5f) gaps.Add("airborne");
        float along = frame["along"];
        bool wide = along <= WideWindowAlong;
        float maxCross = wide ? MathF.Min(CaptureMaxCross, CrossPerRingDistance * (FirstRingAlong - along)) : 220f;
        float maxGlide = wide ? CaptureMaxGlideError : 80f;
        float minSpeed = wide ? CaptureMinSpeed : 37f, maxSpeed = wide ? CaptureMaxSpeed : 65f;
        float minTrackCos = wide ? CaptureMinTrackCos : 0.82f, maxRoll = wide ? CaptureMaxRoll : 18f;
        if (along < -CaptureMaxDistance) gaps.Add($"distance:{-along / 1000f:0.0}/{CaptureMaxDistance / 1000f:0.0}");
        else if (along > -1050f) gaps.Add("too-close");
        if (MathF.Abs(frame["cross"]) > maxCross) gaps.Add($"cross:{frame["cross"]:0}/{maxCross:0}");
        float glide = GlideError(frame);
        if (MathF.Abs(glide) > maxGlide) gaps.Add($"glide:{glide:+0;-0}/{maxGlide:0}");
        if (frame["agl"] < MinCaptureAgl) gaps.Add("low");
        if (frame["speed"] < minSpeed || frame["speed"] > maxSpeed) gaps.Add($"speed:{frame["speed"]:0}/{minSpeed:0}-{maxSpeed:0}");
        if (frame["track_cos"] < minTrackCos) gaps.Add($"track:{MathF.Acos(Math.Clamp(frame["track_cos"], -1f, 1f)) * 57.2958f:0}/{MathF.Acos(minTrackCos) * 57.2958f:0}");
        if (MathF.Abs(frame["roll"]) > maxRoll) gaps.Add($"bank:{MathF.Abs(frame["roll"]):0}/{maxRoll:0}");
        if (frame["stall_warning"] >= 0.5f) gaps.Add("stall");
        if (MathF.Abs(frame["sink"]) > 6f) gaps.Add("vertical-speed");
        return gaps;
    }

    private static bool Unstable(LandingFrame frame)
    {
        if (frame["stall_warning"] >= 0.5f || MathF.Abs(frame["roll"]) > 35f) return true;
        if (frame["agl"] > MinCaptureAgl) return false;
        return frame["gear"] < GearLocked || frame["flap"] < 1.9f || frame["speed"] is < 34f or > 64f ||
            frame["sink"] > 5.0f || MathF.Abs(frame["roll"]) > 15f ||
            MathF.Abs(frame["cross"]) > MathF.Min(18f, frame["runway_half_width"] * 0.8f) ||
            frame["track_cos"] < 0.96f;
    }

    private static bool GoAroundWindow(LandingFrame frame) => frame["agl"] >= MinGoAroundAgl &&
        frame["along"] < frame["runway_length"] * 0.5f && MathF.Abs(frame["cross"]) <= 400f &&
        frame["speed"] >= 35f && frame["sink"] <= 6f && MathF.Abs(frame["roll"]) <= 30f &&
        frame["stall_warning"] < 0.5f && frame["track_cos"] >= 0.7f;

    private static bool EngageableCrew(LandingFrame frame) => frame["other_known"] >= 0.5f
        ? MathF.Abs(frame["other_stick"]) <= 0.12f
        : !frame.Host && frame.RemoteResponseMonitored;

    private static bool Usable(LandingFrame frame)
    {
        if (!frame.Valid || !double.IsFinite(frame.Time) || frame.Time < 0 ||
            (frame.MotionSourceTime is double source && (!double.IsFinite(source) || source < 0))) return false;
        foreach (string feature in Required)
            if (!float.IsFinite(frame[feature])) return false;
        return frame["observation_age"] is >= 0f and <= MaxObservationAge &&
            frame["next_gate_index"] is >= 0f and <= 5f && frame["runway_length"] >= 300f &&
            frame["runway_half_width"] >= 5f && frame["agl"] >= -5f && frame["ground_speed"] >= 0f && frame["speed"] >= 0f &&
            SquaredNorm(frame["track_sin"], frame["track_cos"]) is >= 0.9f and <= 1.1f &&
            SquaredNorm(frame["heading_sin"], frame["heading_cos"]) is >= 0.9f and <= 1.1f;
    }

    private static float SquaredNorm(float first, float second) => first * first + second * second;

    private bool Jumped(LandingFrame frame, double elapsed)
    {
        float horizon = (float)Math.Max(0.05, elapsed);
        float lateralBound = 20f + MathF.Max(frame["ground_speed"], 80f) * horizon;
        return MathF.Abs(frame["along"] - _previousAlong) > lateralBound ||
            MathF.Abs(frame["cross"] - _previousCross) > lateralBound ||
            MathF.Abs(frame["agl"] - _previousAgl) > 12f + 30f * horizon ||
            MathF.Abs(frame["runway_length"] - _previousRunwayLength) > 1f ||
            MathF.Abs(frame["runway_half_width"] - _previousRunwayHalfWidth) > 0.5f;
    }

    private void Remember(LandingFrame frame, bool usable, bool discontinuity)
    {
        double elapsed = frame.Time - _previousTime;
        if (usable && _lastUsable && !discontinuity && elapsed > 0.001)
        {
            float blend = 1f - MathF.Exp((float)(-elapsed / 0.8));
            _speedTrend += blend * ((frame["speed"] - _previousSpeed) / (float)elapsed - _speedTrend);
            _sinkTrend += blend * ((frame["sink"] - _previousSink) / (float)elapsed - _sinkTrend);
            _trendSeconds += elapsed;
            bool demandingPower = frame["throttle"] >= 0.75f || frame["last_throttle"] >= 0.75f;
            _highPowerSeconds = demandingPower ? _highPowerSeconds + elapsed : 0;
        }
        else if (!usable || !_lastUsable || discontinuity)
        {
            _speedTrend = _sinkTrend = 0f;
            _highPowerSeconds = _trendSeconds = 0;
        }
        _hadFrame = true;
        _lastUsable = usable;
        _previousHost = frame.Host;
        _sessionId = frame.SessionId;
        _previousTime = frame.Time;
        _previousMotionTime = usable ? frame.MotionSourceTime : null;
        if (!usable) return;
        _previousAlong = frame["along"];
        _previousCross = frame["cross"];
        _previousAgl = frame["agl"];
        _previousRunwayLength = frame["runway_length"];
        _previousRunwayHalfWidth = frame["runway_half_width"];
        _previousSpeed = frame["speed"];
        _previousSink = frame["sink"];
    }

    private Power AvailablePower(LandingFrame frame)
    {
        if (!KnownEngine(frame, "left") || !KnownEngine(frame, "right")) return Power.Unknown;
        bool left = CanRun(frame, "left");
        bool right = CanRun(frame, "right");
        float delivered = RunningThrust(frame);
        if ((!left && !right) || (_highPowerSeconds >= 1.5 && delivered < 0.15f)) return Power.Incapable;
        if (!left || !right) return Power.SingleEngine;
        if (frame["left_health"] < 0.75f || frame["right_health"] < 0.75f ||
            (_highPowerSeconds >= 1.5 && delivered < 0.55f)) return Power.Derated;
        return Power.Normal;
    }

    private static bool KnownEngine(LandingFrame frame, string side) => frame[side + "_valid"] >= 0.5f &&
        float.IsFinite(frame[side + "_state"]) && frame[side + "_state"] is >= 0f and <= 3f &&
        float.IsFinite(frame[side + "_health"]) && frame[side + "_health"] is >= 0f and <= 1f &&
        float.IsFinite(frame[side + "_thrust"]) && frame[side + "_thrust"] is >= 0f and <= 1.01f &&
        float.IsFinite(frame[side + "_fire"]) && float.IsFinite(frame[side + "_repair"]);

    private static bool CanRun(LandingFrame frame, string side) => frame[side + "_state"] == 2f &&
        frame[side + "_fire"] < 0.5f && frame[side + "_health"] >= 0.25f;

    private static string PowerHint(LandingFrame frame, Power power)
    {
        if (power == Power.Normal) return string.Empty;
        if (power == Power.Unknown) return " Engine capability unknown.";
        string condition = power switch
        {
            Power.Incapable => frame["on_ground"] >= 0.5f ? " Engines unavailable." : " Insufficient engine power.",
            Power.SingleEngine => " Single-engine power.",
            _ => " Engines derated."
        };
        if (frame["left_fire"] >= 0.5f) condition += " Left engine fire.";
        if (frame["right_fire"] >= 0.5f) condition += " Right engine fire.";
        if (frame["left_repair"] is > 0f and < 1f)
            condition += " " + FormattableString.Invariant($"Left repair {frame["left_repair"] * 100:0}%.");
        if (frame["right_repair"] is > 0f and < 1f)
            condition += " " + FormattableString.Invariant($"Right repair {frame["right_repair"] * 100:0}%.");
        return condition;
    }

    private bool SustainableApproach(LandingFrame frame) => frame["speed"] is >= 36f and <= 65f &&
        frame["sink"] <= 6f && MathF.Abs(frame["roll"]) <= 24f && frame["stall_warning"] < 0.5f &&
        RunningThrust(frame) >= 0.12f &&
        (_trendSeconds < 0.8 || (_speedTrend >= -2.8f && _sinkTrend <= 2f));

    private bool ClimbAvailable(LandingFrame frame, Power power) => power == Power.Normal ||
        (power is Power.SingleEngine or Power.Derated && _trendSeconds >= 0.8 &&
        frame["speed"] >= 42f && frame["sink"] <= 3f && MathF.Abs(frame["roll"]) <= 24f &&
        RunningThrust(frame) >= 0.6f && _speedTrend >= -0.8f && _sinkTrend <= 0.8f);

    private static float RunningThrust(LandingFrame frame) =>
        (CanRun(frame, "left") ? frame["left_thrust"] : 0f) + (CanRun(frame, "right") ? frame["right_thrust"] : 0f);

    private static string DescribeUnavailablePower(Power power, string powerHint) =>
        power == Power.Incapable ? "Take control for a forced landing." + powerHint : "Take control." + powerHint;

    private static string Advice(LandingFrame frame, Power power, string standard, bool advisory)
    {
        string powerHint = PowerHint(frame, power);
        if (frame["on_ground"] >= 0.5f)
            return standard + powerHint + (advisory ? " Hold the runway centerline and slow down." : "");
        if (GearJammedInFlight(frame)) return standard + " " + JammedGearCue + powerHint;
        if (power == Power.Incapable) return standard + " " + DescribeUnavailablePower(power, powerHint) + Cue(frame, advisory);
        if (advisory && (frame.GateMissed || Unstable(frame)))
        {
            string reason = frame.GateMissed ? " Ring missed." : " Unstable approach.";
            standard += reason + (power == Power.Normal && GoAroundWindow(frame)
                ? " Go-around advised. Fly manually." : " Take control.");
        }
        return standard + powerHint + Cue(frame, advisory);
    }

    private static string Cue(LandingFrame frame, bool advisory)
    {
        string hazards = string.Empty;
        if (frame["ice"] >= 0.3f) hazards += " Wing ice.";
        if (frame["turbulence"] >= 0.6f) hazards += " Heavy turbulence.";
        if (MathF.Abs(frame["wind_cross"]) >= 10f) hazards += " Strong crosswind.";
        if (!advisory) return hazards + " Outside approach area.";
        if (frame["gear"] < GearLocked && frame["agl"] < FinalAgl) hazards += " Check landing gear.";
        if (frame["along"] > 0f) return hazards;

        float cross = frame["cross"];
        float glideError = frame["agl"] + (frame["along"] - 250f) * 0.052408f;
        float suggestedSpeed = 47f + Math.Clamp(frame["ice"], 0f, 1f) * 6f + Math.Clamp(frame["turbulence"], 0f, 1f) * 4f;
        string lateral = MathF.Abs(cross) < 4f ? "centerline" : FormattableString.Invariant($"{MathF.Abs(cross):0} m {(cross > 0f ? "right" : "left")}");
        string vertical = MathF.Abs(glideError) < 4f ? "on glide path" : FormattableString.Invariant($"{MathF.Abs(glideError):0} m {(glideError > 0f ? "high" : "low")}");
        if (frame["next_gate_index"] < 5 && frame["next_gate_distance"] is > 0f and < 350f)
            vertical = FormattableString.Invariant($"{MathF.Abs(frame["gate_height_error"]):0} m {(frame["gate_height_error"] >= 0f ? "above" : "below")} next ring");
        return hazards + FormattableString.Invariant($" {lateral}, {vertical}, {frame["speed"]:0}/{suggestedSpeed:0} m/s.");
    }

    private static bool InApproachArea(LandingFrame frame) =>
        frame["along"] >= -9000f && frame["along"] <= frame["runway_length"] &&
        MathF.Abs(frame["cross"]) <= 2000f && frame["agl"] <= 900f;

    private static bool InLandingContext(LandingFrame frame, LandingRouteStage stage)
    {
        if (stage == LandingRouteStage.Other || !InApproachArea(frame)) return false;
        bool arrivalConfirmed = stage == LandingRouteStage.Rollout || frame.GateMask != 0;
        if (frame["on_ground"] >= 0.5f)
            return arrivalConfirmed && OnRunway(frame) && frame["ground_speed"] > StopSpeed;
        if (stage == LandingRouteStage.Approach) return true;
        return frame["track_cos"] >= 0.5f && (frame["along"] < 0f || arrivalConfirmed) && frame["sink"] >= -3f;
    }
}
