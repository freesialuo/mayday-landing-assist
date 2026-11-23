using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MaydayLanding;

/// <summary>One header per recording and one decision per line. Local engine sampling and network
/// source time are separate: a null source time means the game exposed no source stamp.</summary>
public sealed class LandingTrace : IDisposable
{
    private readonly StreamWriter _writer;
    private long _sequence;
    private static readonly JsonSerializerOptions Options = new() { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };

    /// <summary>Recording file; a probe summary is written beside it as *.summary.json.</summary>
    public string Path { get; }

    private LandingTrace(string path, string? policySha256, string? probe)
    {
        Path = path;
        _writer = new StreamWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read),
            new UTF8Encoding(false));
        _writer.WriteLine(JsonSerializer.Serialize(new { schema = "mayday-landing.trace/1", features = LandingFeatures.Names,
            actions = new[] { "pitch", "roll", "yaw", "throttle", "brake" }, utc = DateTime.UtcNow, policySha256, probe },
            Options));
        _writer.Flush();
    }

    public static LandingTrace Open(string directory, string? policySha256 = null, string? probe = null)
    {
        Directory.CreateDirectory(directory);
        return new LandingTrace(System.IO.Path.Combine(directory,
            $"landing-{DateTime.UtcNow:yyyyMMdd-HHmmss-fff}-{Guid.NewGuid():N}.jsonl"), policySha256, probe);
    }

    public void WriteSummary(object summary) =>
        File.WriteAllText(System.IO.Path.ChangeExtension(Path, ".summary.json"),
            JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }), new UTF8Encoding(false));

    private static double? Finite(double? value) => value.HasValue && double.IsFinite(value.Value) ? value : null;
    private static float? FiniteFloat(float value) => float.IsFinite(value) ? value : null;
    private static float?[]? Samples(float[]? values) => values?.Select(FiniteFloat).ToArray();

    public void Write(LandingFrame frame, int phase, string status, bool authorized, float[]? proposed, float[]? sent,
        object? probe = null)
    {
        _writer.WriteLine(JsonSerializer.Serialize(new
        {
            seq = _sequence++, time = Finite(frame.Time), session = frame.SessionId, host = frame.Host,
            valid = frame.Valid, seated = frame.Seated, ap = frame.NativeAp, pilotOverride = frame.PilotOverride,
            otherOverride = frame.OtherPilotOverride, externalController = frame.ExternalController, crashed = frame.Crashed,
            missed = frame.GateMissed, sourceTime = Finite(frame.MotionSourceTime),
            receivedAt = Finite(frame.MotionReceivedAt), engineSourceTime = Finite(frame.EngineSourceTime),
            phase, status, authorized, axes = frame.AuthorizedAxes, gates = frame.GateMask,
            responseMonitored = frame.RemoteResponseMonitored,
            predictedNow = frame.PredictedNow, policyReset = frame.PolicyReset,
            leftFaults = frame.LeftEngineFaults, rightFaults = frame.RightEngineFaults,
            leftLever = FiniteFloat(frame.LeftLever), rightLever = FiniteFloat(frame.RightLever),
            parkingBrake = frame.ParkingBrakeSet, gearJammed = frame.GearJammed, runwayLatched = frame.RunwayLatched, reverse = frame.ReverseOn,
            throttleSent = Finite(frame.ThrottleSent), parkingBrakeRequested = frame.ParkingBrakeRequested,
            observation = Samples(frame.Values), pilot = Samples(frame.RawInputs),
            proposed = Samples(proposed), sent = Samples(sent), touchdownScore = Finite(frame.TouchdownScore), probe
        }));
        if (_sequence % 30 == 0 || frame.Crashed || frame.PilotOverride) _writer.Flush();
    }

    public void Dispose() => _writer.Dispose();
}
