using System;
using System.Collections.Generic;
using System.Linq;

namespace MaydayLanding;

/// <summary>Native disturbance fields read on the host (server-side PlaneFlight state).</summary>
public readonly record struct ProbeNative(bool PendingGust, float GustDir, float GustT, float HazardTurb,
    float JoltT, float JoltAccel);

/// <summary>Test-build probe: per-frame extras for the trace plus a per-recording summary.
/// Gameplay probes check twin fidelity (gust kick, touchdown outcome, delay); model probes
/// compare the policy with the observable prior and with the pilot's own inputs.</summary>
public sealed class LandingProbe
{
    public const string Schema = "mayday-landing.probe/1";
    private static readonly string[] Axes = { "pitch", "roll", "yaw", "throttle" };
    private readonly List<Dictionary<string, object?>> _pending = new();
    private readonly List<Dictionary<string, object?>> _touchdowns = new();
    private readonly Dictionary<string, int> _hazards = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _releases = new(StringComparer.Ordinal);
    private readonly double[] _priorGap = new double[4], _pilotGap = new double[4];
    private int _priorSamples, _pilotSamples, _frames, _gusts, _marks, _engagements;
    private double _engagedSeconds, _firstTime = double.NaN, _lastTime = double.NaN, _maxAge;
    private float _lastGustT;
    private bool _lastAuthorized;
    private bool _wasAirborne;
    private float _lastSink, _lastRoll, _lastAlong, _lastCross, _lastSpeed, _lastGear;
    private string? _lastStatus;

    /// <summary>Queue a discrete event (touchdown, hazard, pilot mark) for the next trace line.</summary>
    public void Event(string type, double time, Dictionary<string, object?>? data = null)
    {
        var entry = new Dictionary<string, object?>(data ?? new Dictionary<string, object?>()) { ["type"] = type, ["time"] = Finite(time) };
        _pending.Add(entry);
        switch (type)
        {
            case "touchdown":
            case "contact": _touchdowns.Add(entry); break;
            case "hazard": Count(_hazards, entry.TryGetValue("kind", out var kind) ? kind?.ToString() ?? "?" : "?"); break;
            case "mark": _marks++; break;
        }
    }

    /// <summary>Record one decision frame; returns the probe object for the trace line.</summary>
    public Dictionary<string, object?> Observe(LandingFrame frame, ProbeNative? native, float[]? proposal, float[]? prior,
        bool authorized, string status)
    {
        double time = frame.Time;
        if (double.IsFinite(time))
        {
            if (double.IsNaN(_firstTime)) _firstTime = time;
            if (_lastAuthorized && double.IsFinite(_lastTime) && time > _lastTime && time - _lastTime < 1.0)
                _engagedSeconds += time - _lastTime;
            _lastTime = time;
        }
        _frames++;
        if (frame.Valid) _maxAge = Math.Max(_maxAge, frame["observation_age"]);
        if (authorized && !_lastAuthorized) _engagements++;
        if (!authorized && _lastAuthorized) Count(_releases, status);
        _lastAuthorized = authorized;

        if (native is { } n)
        {
            // gustT is set to 0.3 s when a pending front gust is applied (ApplyTurbulence).
            if (n.GustT > 0f && _lastGustT <= 0f)
            {
                _gusts++;
                Event("gust", time, new Dictionary<string, object?>
                {
                    ["dir"] = n.GustDir, ["roll"] = frame["roll"], ["agl"] = frame["agl"], ["onGround"] = frame["on_ground"] >= 0.5f
                });
            }
            _lastGustT = n.GustT;
        }

        if (frame.Valid)
        {
            // Contact from our own frames: the RpcTouchdown hook does not fire on the host.
            bool airborne = frame["on_ground"] < 0.5f;
            if (_wasAirborne && !airborne)
                Event("contact", time, new Dictionary<string, object?>
                {
                    ["sink"] = _lastSink, ["roll"] = _lastRoll, ["along"] = _lastAlong, ["cross"] = _lastCross,
                    ["speed"] = _lastSpeed, ["gear"] = _lastGear, ["authorized"] = authorized,
                    // Native OnTouchdown: crash above 8 m/s or 30 deg; HardLanding hazard at >= 2.5 m/s or gear up;
                    // wing-strike damage above WingStrikeRoll (9.06 deg in the recorded tuning).
                    ["nativeCrash"] = _lastSink > 8f || Math.Abs(_lastRoll) > 30f,
                    ["nativeHardLanding"] = _lastSink >= 2.5f || _lastGear < 0.9f,
                    ["wingStrike"] = Math.Abs(_lastRoll) > 9.06f
                });
            if (airborne)
            {
                _lastSink = frame["sink"]; _lastRoll = frame["roll"]; _lastAlong = frame["along"];
                _lastCross = frame["cross"]; _lastSpeed = frame["speed"]; _lastGear = frame["gear"];
            }
            _wasAirborne = airborne;
        }

        if (proposal != null && prior != null)
        {
            for (int i = 0; i < 4; i++) _priorGap[i] += Math.Abs(proposal[i] - prior[i]);
            _priorSamples++;
        }
        var raw = frame.RawInputs;
        if (!authorized && proposal != null && raw != null && raw.Length >= 4 && !frame.NativeAp)
        {
            // Raw throttle is the lever axis; compare it with the throttle feature instead.
            for (int i = 0; i < 3; i++) _pilotGap[i] += Math.Abs(proposal[i] - raw[i]);
            _pilotGap[3] += Math.Abs(proposal[3] - frame["throttle"]);
            _pilotSamples++;
        }

        var line = new Dictionary<string, object?>
        {
            ["prior"] = prior?.Select(Finite).ToArray(),
            ["native"] = native is { } v ? new Dictionary<string, object?>
            {
                ["pendingGust"] = v.PendingGust, ["gustDir"] = v.GustDir, ["gustT"] = v.GustT,
                ["hazardTurb"] = v.HazardTurb, ["joltT"] = v.JoltT, ["joltAccel"] = v.JoltAccel
            } : null,
            ["events"] = _pending.Count > 0 ? _pending.ToArray() : null,
            ["statusChanged"] = status != _lastStatus ? status : null,
        };
        _lastStatus = status;
        _pending.Clear();
        return line;
    }

    public Dictionary<string, object?> Summary(string? policySha256, string mode)
    {
        return new Dictionary<string, object?>
        {
            ["schema"] = Schema, ["mode"] = mode, ["policySha256"] = policySha256, ["utc"] = DateTime.UtcNow,
            ["frames"] = _frames, ["seconds"] = double.IsFinite(_lastTime - _firstTime) ? _lastTime - _firstTime : 0,
            ["engagements"] = _engagements, ["engagedSeconds"] = _engagedSeconds,
            ["releases"] = new Dictionary<string, int>(_releases),
            ["maxObservationAge"] = _maxAge, ["gustOnsets"] = _gusts, ["marks"] = _marks,
            ["hazards"] = new Dictionary<string, int>(_hazards), ["touchdowns"] = _touchdowns.ToArray(),
            ["meanAbsPolicyMinusPrior"] = Means(_priorGap, _priorSamples),
            ["meanAbsPolicyMinusPilot"] = Means(_pilotGap, _pilotSamples),
            ["samples"] = new Dictionary<string, int> { ["prior"] = _priorSamples, ["pilot"] = _pilotSamples },
        };
    }

    private static Dictionary<string, double>? Means(double[] sums, int count) =>
        count == 0 ? null : Axes.Select((axis, i) => (axis, value: sums[i] / count)).ToDictionary(x => x.axis, x => x.value);

    private static void Count(Dictionary<string, int> counts, string key) =>
        counts[key] = counts.TryGetValue(key, out int value) ? value + 1 : 1;

    private static double? Finite(double value) => double.IsFinite(value) ? value : null;
    private static float? Finite(float value) => float.IsFinite(value) ? value : null;
}
