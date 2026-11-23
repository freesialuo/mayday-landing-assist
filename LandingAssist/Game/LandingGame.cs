// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using Il2CppMayday;
using Il2CppMayday.Airport;
using Il2CppMayday.Flight;
using Il2CppMayday.Systems;
using UnityEngine;

namespace MaydayLanding;

internal readonly record struct PilotSample(float Pitch, float Roll, float Yaw, float Throttle, bool Brake)
{
    public static PilotSample Read() => new(GameInput.Pitch, GameInput.Roll, GameInput.Yaw,
        GameInput.ThrottleAxis, GameInput.BrakesHeld);

    public bool Takeover => !(float.IsFinite(Pitch) && float.IsFinite(Roll) && float.IsFinite(Yaw)
        && float.IsFinite(Throttle)) || MathF.Abs(Pitch) > 0.2f || MathF.Abs(Roll) > 0.2f
        || MathF.Abs(Yaw) > 0.25f || MathF.Abs(Throttle) > 0.15f || Brake;
}

internal sealed class LandingGame
{
    internal const float StaleTime = 999f;
    private const int GateCount = 5;
    private const float GateVisualHeight = 2.8f;
    private (IntPtr Plane, uint Player, int Seat, int Seed, int Leg, float RunStart, int Destination) _identity;
    private long _session;
    private double _latestStamp = double.NaN;
    private double _lastAdvance;
    private bool _sawGateMiss;
    // Last valid runway, kept so an engaged rollout can finish after the game's nav leaves the runway leg.
    private (Vector3 Direction, Vector3 Threshold, float Length, float Width, long Session)? _runway;

    /// <summary>Set by the mod while it owns an engaged rollout; allows the latched runway to stand in.</summary>
    public bool HoldRunway { get; set; }

    public PlaneFlight? Plane { get; private set; }
    public uint PlayerId { get; private set; }
    public int Seat { get; private set; }
    public bool OtherOccupied { get; private set; }
    public int PassedGates { get; private set; }
    public int MissedGates { get; private set; }
    public string GradeKey { get; private set; } = "";
    public float GradeStars { get; private set; }

    public LandingFrame Read(bool externalController)
    {
        double now = Time.unscaledTimeAsDouble;
        var frame = new LandingFrame { Time = now, ExternalController = externalController };
        frame["observation_age"] = StaleTime;
        frame["next_gate_index"] = GateCount;
        GradeKey = "";
        GradeStars = 0;
        OtherOccupied = false;
        PassedGates = MissedGates = 0;

        var plane = PlaneFlight.Instance;
        Plane = plane;
        var me = PlayerNet.Local;
        var game = GameState.Instance;
        PlayerId = me == null ? 0 : me.netId;
        Seat = me == null ? 0 : me.IsPiloting ? 1 : PlaneFlight.IsCopilot(me) ? 2 : 0;
        int destination = game == null ? -1 : game.NetworkDestinationAirport;
        var identity = (plane == null ? IntPtr.Zero : plane.Pointer, PlayerId, Seat,
            game == null ? 0 : game.NetworkRunSeed, game == null ? 0 : game.NetworkLeg,
            game == null ? 0f : game.NetworkRunStartTime, destination);
        if (identity != _identity)
        {
            _identity = identity;
            _session++;
            _latestStamp = double.NaN;
            _lastAdvance = now;
            _sawGateMiss = false;
        }
        frame.SessionId = _session;
        frame.Seated = Seat != 0 && PlayerId != 0 && Application.isFocused && !UIState.BlocksGameplay;
        if (plane == null || !plane.HasRender) return frame;

        frame.Host = plane.isServer;
        frame.Seated &= frame.Host || Il2CppMirror.NetworkClient.isConnected;
        frame.NativeAp = (plane.NetworkapModes & 1) != 0;
        frame.Crashed = plane.Networkcrashed;
        frame.GearJammed = plane.NetworkgearJammed;
        frame.ParkingBrakeSet = plane.NetworkparkingBrake;
        frame.ReverseOn = plane.NetworkreverseGear;
        if (frame.Seated)
        {
            var manual = PilotSample.Read();
            frame.PilotOverride = manual.Takeover || plane.LocalHandOnThrottle;
            if (!externalController && float.IsFinite(manual.Pitch) && float.IsFinite(manual.Roll)
                && float.IsFinite(manual.Yaw) && float.IsFinite(manual.Throttle))
                frame.RawInputs = new[] { manual.Pitch, manual.Roll, manual.Yaw, manual.Throttle, manual.Brake ? 1f : 0f };
            ReadOtherSeat(frame, plane, Seat);
        }
        var director = GameDirector.Instance;
        if (director != null)
        {
            GradeKey = director.NetworklastGradeKey ?? "";
            GradeStars = FiniteOrZero(director.NetworklastStars);
        }

        var nav = plane.Nav;
        var direction = new Vector3(nav.dir.x, 0, nav.dir.z);
        Vector3 threshold = nav.thresholdA;
        float runwayLength = nav.runwayLength;
        float width = 0f;
        bool navUsable = nav.valid && float.IsFinite(direction.x) && float.IsFinite(direction.z)
            && nav.runwayLength > 100f && direction.sqrMagnitude >= 0.01f;
        if (navUsable)
        {
            var airport = destination >= 0 ? AirportSites.Find(destination) : null;
            if (airport == null && plane.navRoute != null) airport = plane.navRoute.lastDest;
            width = airport == null ? 0 : airport.runway.width;
            navUsable = width > 0f && float.IsFinite(width);
            if (navUsable) _runway = (direction, threshold, runwayLength, width, _session);
        }
        if (!navUsable)
        {
            if (!HoldRunway || _runway is not { } latched || latched.Session != _session) return frame;
            (direction, threshold, runwayLength, width, _) = latched;
            frame.RunwayLatched = true;
        }

        var rendered = plane.Render;
        var state = frame.Host ? plane.physCurr : rendered;
        frame["observation_age"] = ObservationAge(plane, state, frame.Host, now);
        if (frame["observation_age"] == StaleTime) return frame;
        frame.MotionSourceTime = state.time;
        frame.MotionReceivedAt = _lastAdvance;

        direction.Normalize();
        var right = new Vector3(direction.z, 0, -direction.x);
        var relative = state.pos - threshold;
        var groundVelocity = new Vector3(state.vel.x, 0, state.vel.z);
        var sim = frame.Host ? plane.sim : null;
        var wind = sim != null ? sim.EffectiveWind : state.Wind;
        float course = Mathf.Atan2(direction.x, direction.z);
        float track = (groundVelocity.sqrMagnitude > 9f
            ? Mathf.Atan2(groundVelocity.x, groundVelocity.z) : state.heading * Mathf.Deg2Rad) - course;
        float heading = state.heading * Mathf.Deg2Rad - course;

        frame["along"] = Vector3.Dot(relative, direction);
        frame["cross"] = Vector3.Dot(relative, right);
        frame["agl"] = sim != null ? sim.Agl : PlaneTelemetry.AGL;
        frame["speed"] = sim != null ? sim.Speed : state.AirVelocity.magnitude;
        frame["ground_speed"] = groundVelocity.magnitude;
        frame["sink"] = -state.vel.y;
        frame["pitch"] = state.pitch;
        frame["roll"] = state.roll;
        frame["pitch_rate"] = state.pitchRate;
        frame["roll_rate"] = state.rollRate;
        frame["yaw_rate"] = state.yawRate;
        frame["track_sin"] = Mathf.Sin(track);
        frame["track_cos"] = Mathf.Cos(track);
        frame["heading_sin"] = Mathf.Sin(heading);
        frame["heading_cos"] = Mathf.Cos(heading);
        frame["wind_along"] = Vector3.Dot(wind, direction);
        frame["wind_cross"] = Vector3.Dot(wind, right);
        frame["runway_length"] = runwayLength;
        frame["runway_half_width"] = width * 0.5f;
        frame["gear"] = state.Gear01;
        frame["flap"] = state.Flap02;
        frame["stall_warning"] = (state.flags & 4) != 0 || PlaneTelemetry.Stalling ? 1 : 0;
        frame["on_ground"] = (state.flags & 1) != 0 ? 1 : 0;
        frame.Crashed |= (state.flags & 0x80) != 0;
        frame.LeftLever = state.ThrottleL;
        frame.RightLever = state.ThrottleR;
        frame["throttle"] = (frame.LeftLever + frame.RightLever) * 0.5f;
        frame.LeftEngineFaults = ReadEngine(frame, 0, "left");
        frame.RightEngineFaults = ReadEngine(frame, 1, "right");
        frame["rain"] = WorldWeather.Rain;
        frame["visibility"] = WorldWeather.Valid ? WorldWeather.Visibility : WorldWeather.ClearVisibility;
        frame["turbulence"] = PlaneTelemetry.Turbulence;
        frame["ice"] = PlaneTelemetry.WingIce;
        if (!ReadGates(frame, state.pos, direction)) return frame;

        frame.Valid = true;
        foreach (float value in frame.Values)
            if (!float.IsFinite(value)) { frame.Valid = false; break; }
        return frame;
    }

    private float ObservationAge(PlaneFlight plane, FlightSnapshot state, bool host, double now)
    {
        if (!double.IsFinite(state.time) || state.time < 0d) return StaleTime;
        double latest = state.time;
        double poseLag = 0d;
        double transitLag = 0d;
        if (!host)
        {
            var buffer = plane.buffer;
            if (buffer == null || !buffer.clockInit || buffer.count <= 0) return StaleTime;
            var items = buffer.items;
            if (items == null || buffer.count > items.Length) return StaleTime;
            var newest = items[buffer.count - 1];
            latest = newest.time;
            if (!double.IsFinite(latest) || latest < 0d) return StaleTime;
            double rtt = Il2CppMirror.NetworkTime.rtt;
            if (!double.IsFinite(rtt) || rtt < 0d || newest.rb != state.rb) return StaleTime;
            // Compare only native source stamps with each other. Arrival and decision timing share the local clock.
            poseLag = Math.Max(0d, latest - state.time);
            transitLag = rtt * 0.5d;
        }
        if (_latestStamp != latest)
        {
            bool rewound = double.IsFinite(_latestStamp) && latest < _latestStamp;
            _latestStamp = latest;
            _lastAdvance = now;
            if (rewound) return StaleTime;
        }
        double age = poseLag + transitLag + Math.Max(0d, now - _lastAdvance);
        return double.IsFinite(age) && age < StaleTime ? (float)age : StaleTime;
    }

    private static ushort ReadEngine(LandingFrame frame, int index, string prefix)
    {
        var unit = SysRefs.Engine(index);
        if (unit == null) return 0;
        float rpm = unit.Rpm, thrust = unit.Thrust, health = unit.Health;
        float fire = unit.FireIntensity, repair = unit.RepairProgress;
        int state = (byte)unit.State;
        if (!float.IsFinite(rpm) || !float.IsFinite(thrust) || !float.IsFinite(health)
            || !float.IsFinite(fire) || !float.IsFinite(repair) || state > 3) return unit.Faults;

        frame[prefix + "_rpm"] = rpm;
        frame[prefix + "_thrust"] = thrust;
        frame[prefix + "_health"] = health;
        frame[prefix + "_fire"] = fire;
        frame[prefix + "_repair"] = repair;
        frame[prefix + "_state"] = state;
        frame[prefix + "_valid"] = 1;
        return unit.Faults;
    }

    private static float FiniteOrZero(float value) => float.IsFinite(value) ? value : 0f;

    internal bool ReadOtherSeat(LandingFrame frame, PlaneFlight plane, int seat)
    {
        frame["other_known"] = 0;
        frame["other_stick"] = 0;
        if (CockpitIntrusion.Seated)
        {
            frame.OtherPilotOverride = true;
            return true;
        }
        var other = seat == 1 ? PlaneFlight.FindCopilot() : PlaneFlight.FindPilot();
        OtherOccupied = other != null;
        if (other == null) { frame["other_known"] = 1; return false; }
        if (!plane.isServer) return false;

        frame["other_known"] = 1;
        float age = Time.time - (seat == 1 ? plane.cpTime : plane.inTime);
        uint sender = seat == 1 ? plane.cpPilot : plane.inPilot;
        if (sender != other.netId || !(age is >= 0f and < 0.4f)) return false;
        float pitch = seat == 1 ? plane.cpPitch : plane.inPitch;
        float roll = seat == 1 ? plane.cpRoll : plane.inRoll;
        float yaw = seat == 1 ? plane.cpYaw : plane.inYaw;
        float throttle = seat == 1 ? plane.cpThrottle : plane.inThrottle;
        bool brake = seat == 1 ? plane.cpBrakes : plane.inBrakes;
        bool takeover = new PilotSample(pitch, roll, yaw, throttle, brake).Takeover;
        if (!(float.IsFinite(pitch) && float.IsFinite(roll) && float.IsFinite(yaw)))
            frame["other_known"] = 0;
        else frame["other_stick"] = MathF.Max(MathF.Abs(pitch), MathF.Max(MathF.Abs(roll), MathF.Abs(yaw)));
        frame.OtherPilotOverride = takeover;
        return takeover;
    }

    private bool ReadGates(LandingFrame frame, Vector3 position, Vector3 direction)
    {
        var states = NavTelemetry.GateState;
        var centers = NavTelemetry.GateCenter;
        var halves = NavTelemetry.GateHalf;
        if (NavTelemetry.GateCount != GateCount || states == null || states.Length < GateCount
            || centers == null || centers.Length < GateCount || halves == null || halves.Length < GateCount)
        {
            frame.GateMissed = _sawGateMiss;
            return (NavTelemetry.Leg == NavLeg.Rollout || frame.RunwayLatched) && frame["on_ground"] > 0.5f;
        }

        int next = GateCount;
        for (int i = 0; i < GateCount; i++)
        {
            byte gate = states[i];
            if (gate > 2) return false;
            frame.GateMask |= gate << (2 * i);
            if (gate == 1) PassedGates++;
            else if (gate == 2) MissedGates++;
            else if (next == GateCount) next = i;
        }
        _sawGateMiss |= MissedGates > 0;
        frame.GateMissed = _sawGateMiss;
        frame["next_gate_index"] = next;
        if (next == GateCount) return true;
        var center = centers[next];
        var half = halves[next];
        frame["next_gate_distance"] = Vector3.Dot(center - position, direction);
        frame["gate_height_error"] = position.y - (center.y - GateVisualHeight);
        frame["gate_half_width"] = half.x;
        frame["gate_half_height"] = half.y;
        return half.x > 0 && half.y > 0;
    }
}
