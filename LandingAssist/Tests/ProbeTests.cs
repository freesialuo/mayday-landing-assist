// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace MaydayLanding.Tests;

internal static class ProbeTests
{
    public static void Run()
    {
        var probe = new LandingProbe();
        var frame = new LandingFrame { Valid = true, Time = 10.0, RawInputs = new[] { 0.1f, -0.2f, 0f, 0.5f, 0f } };
        frame["observation_age"] = 0.2f;
        frame["roll"] = 1.5f;
        frame["agl"] = 6f;
        frame["throttle"] = 0.4f;
        var proposal = new[] { 0.2f, 0.0f, 0.1f, 0.5f, 0f };
        var prior = new[] { 0.1f, 0.1f, 0.1f, 0.4f, 0f };

        var line = probe.Observe(frame, new ProbeNative(false, 1f, 0f, 0f, 0f, 0f), proposal, prior, false, "Final approach.");
        Check(line["events"] == null && line["statusChanged"] as string == "Final approach.", "First frame has no events");

        frame.Time = 10.1;
        probe.Event("touchdown", 10.1, new Dictionary<string, object?> { ["vs"] = 2.2f, ["roll"] = 3f, ["nativeCrash"] = false });
        line = probe.Observe(frame, new ProbeNative(false, -1f, 0.3f, 0.5f, 0f, 0f), proposal, prior, true, "Final approach.");
        var events = (Dictionary<string, object?>[])line["events"]!;
        Check(events.Length == 2 && events.Any(e => (string)e["type"]! == "gust") && events.Any(e => (string)e["type"]! == "touchdown"),
            "Gust onset is detected from gustT and queued events are attached once");
        Check(line["statusChanged"] == null, "Unchanged status is not repeated");

        frame.Time = 10.2;
        line = probe.Observe(frame, new ProbeNative(false, -1f, 0.2f, 0.5f, 0f, 0f), null, null, false, "Pilot has control.");
        Check(line["events"] == null, "A continuing gust is not a new onset");

        var summary = probe.Summary("abc", "auto");
        Check((int)summary["gustOnsets"]! == 1 && (int)summary["engagements"]! == 1, "Summary counts gusts and engagements");
        var releases = (Dictionary<string, int>)summary["releases"]!;
        Check(releases["Pilot has control."] == 1, "Summary counts release reasons");
        var gaps = (Dictionary<string, double>)summary["meanAbsPolicyMinusPrior"]!;
        Check(Math.Abs(gaps["pitch"] - 0.1) < 1e-5 && Math.Abs(gaps["throttle"] - 0.1) < 1e-5, "Policy-prior gap is averaged");
        var pilot = (Dictionary<string, double>)summary["meanAbsPolicyMinusPilot"]!;
        Check(Math.Abs(pilot["roll"] - 0.2) < 1e-5, "Pilot comparison uses only unauthorized frames");

        string directory = Path.Combine(Path.GetTempPath(), "mayday-probe-test-" + Guid.NewGuid().ToString("N"));
        string path;
        using (var trace = LandingTrace.Open(directory, "abc", LandingProbe.Schema))
        {
            trace.Write(frame, 1, "Final approach.", false, proposal, null, line);
            trace.WriteSummary(summary);
            path = trace.Path;
        }
        var lines = File.ReadAllLines(path);
        var header = JsonDocument.Parse(lines[0]).RootElement;
        Check(header.GetProperty("probe").GetString() == LandingProbe.Schema, "Header names the probe schema");
        var row = JsonDocument.Parse(lines[1]).RootElement;
        Check(row.TryGetProperty("sent", out var sent) && sent.ValueKind == JsonValueKind.Null,
            "Trace rows keep null fields for existing readers");
        Check(File.Exists(Path.ChangeExtension(path, ".summary.json")), "Summary is written beside the trace");
        Console.WriteLine("PASS landing probe");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException("Probe: " + message);
    }
}
