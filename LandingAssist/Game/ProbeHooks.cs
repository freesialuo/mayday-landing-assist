#if LANDING_AUTO
using System;
using System.Collections.Generic;
using HarmonyLib;
using Il2CppMayday;
using Il2CppMayday.Flight;

namespace MaydayLanding;

/// <summary>Test-build observers on native touchdown and hazard events. Postfixes only read
/// their arguments and queue an event; they never change game state or control inputs.</summary>
internal static class ProbeHooks
{
    private static readonly string[] HazardNames =
        { "BirdStrike", "Lightning", "Hail", "Icing", "Turbulence", "SevereGust", "Collision", "HardLanding" };
    private static readonly Queue<(string Type, Dictionary<string, object?> Data)> Pending = new();
    private static bool _installed;

    public static string? Install(HarmonyLib.Harmony harmony)
    {
        if (_installed) return null;
        try
        {
            var touchdown = AccessTools.Method(typeof(PlaneFlight),
                "UserCode_RpcTouchdown__Single__Single__Byte__Boolean__Single__Double");
            var hazard = AccessTools.Method(typeof(PlaneFlight), "OnHazard");
            if (touchdown == null || hazard == null) return "probe hooks not found in this game build";
            harmony.Patch(touchdown, postfix: new HarmonyMethod(typeof(ProbeHooks), nameof(AfterTouchdown)));
            harmony.Patch(hazard, postfix: new HarmonyMethod(typeof(ProbeHooks), nameof(AfterHazard)));
            _installed = true;
            return null;
        }
        catch (Exception error)
        {
            return "probe hooks unavailable: " + error.Message;
        }
    }

    /// <summary>Drain queued native events into the probe, stamped with the current frame time.</summary>
    public static void Drain(LandingProbe probe, double time)
    {
        lock (Pending)
            while (Pending.Count > 0)
            {
                var (type, data) = Pending.Dequeue();
                probe.Event(type, time, data);
            }
    }

    private static void Enqueue(string type, Dictionary<string, object?> data)
    {
        lock (Pending)
        {
            if (Pending.Count < 256) Pending.Enqueue((type, data));
        }
    }

    private static void AfterTouchdown(float vs, float roll, byte surface, bool gearDown, float drift, double at)
    {
        try
        {
            // Native OnTouchdown: crash above 8 m/s sink or 30 deg bank; wing strike damage above WingStrikeRoll.
            Enqueue("touchdown", new Dictionary<string, object?>
            {
                ["vs"] = vs, ["roll"] = roll, ["surface"] = surface, ["gearDown"] = gearDown, ["drift"] = drift,
                ["serverTime"] = at, ["nativeCrash"] = vs > 8f || Math.Abs(roll) > 30f
            });
        }
        catch { }
    }

    private static void AfterHazard(PlaneFlight __instance, HazardType type, float param)
    {
        try
        {
            int kind = (int)type;
            Enqueue("hazard", new Dictionary<string, object?>
            {
                ["kind"] = kind >= 0 && kind < HazardNames.Length ? HazardNames[kind] : kind.ToString(),
                ["param"] = param, ["gustDir"] = __instance.gustDir, ["pendingGust"] = __instance.pendingGust
            });
        }
        catch { }
    }
}
#endif
