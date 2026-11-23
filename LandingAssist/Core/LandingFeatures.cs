// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MaydayLanding;

public static class LandingFeatures
{
    public static readonly string[] Names = ReadNames();
    private static readonly Dictionary<string, int> Indices = Names.Select((name, index) => (name, index))
        .ToDictionary(entry => entry.name, entry => entry.index, StringComparer.Ordinal);

    public static int Index(string name) => Indices[name];

    private static string[] ReadNames()
    {
        using var resource = typeof(LandingFeatures).Assembly.GetManifestResourceStream("MaydayLanding.Features")
            ?? throw new InvalidDataException("Missing landing feature schema.");
        using var reader = new StreamReader(resource);
        return reader.ReadToEnd().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    }
}

public sealed class LandingFrame
{
    public double Time;
    public long SessionId;
    public bool Valid, Seated, Host, NativeAp, PilotOverride, OtherPilotOverride, Crashed, GateMissed, ExternalController;
    public double? MotionSourceTime, MotionReceivedAt, EngineSourceTime;
    public float[]? RawInputs;
    public int AuthorizedAxes, GateMask;
    public bool RemoteResponseMonitored;
    public bool PredictedNow, PolicyReset, ParkingBrakeSet, GearJammed;
    public ushort LeftEngineFaults, RightEngineFaults;
    public float LeftLever, RightLever;
    public float? TouchdownScore;
    public float? ThrottleSent;
    public bool ParkingBrakeRequested;
    public bool RunwayLatched, ReverseOn;
    public float[] Values { get; } = new float[LandingFeatures.Names.Length];
    public float this[string name]
    {
        get => Values[LandingFeatures.Index(name)];
        set => Values[LandingFeatures.Index(name)] = value;
    }
}
