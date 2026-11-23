#if LANDING_AUTO
using System;
using System.Reflection;
using HarmonyLib;
using Il2CppMayday;
using Il2CppMayday.Flight;
using UnityEngine;
using HarmonyInstance = HarmonyLib.Harmony;

namespace MaydayLanding;

internal static class LandingInput
{
    internal static readonly MethodInfo[] Getters =
    {
        Getter(nameof(GameInput.Pitch)), Getter(nameof(GameInput.Roll)), Getter(nameof(GameInput.Yaw)),
        Getter(nameof(GameInput.ThrottleAxis)), Getter(nameof(GameInput.BrakesHeld))
    };
    internal static readonly MethodInfo PilotInputMethod = AccessTools.Method(typeof(PlaneFlight), "ClientPilotInput")
        ?? throw new MissingMethodException(typeof(PlaneFlight).FullName, "ClientPilotInput");

    [ThreadStatic] private static bool _readingForFlight;
    private static Func<PlaneFlight, bool>? _stillAuthorized;
    private static Lease? _lease;
    private static bool _manualRelease;
    private static bool _authorityLost;

    private sealed record Lease(IntPtr Plane, double Until, LandingAxes Axes, float[] Action);

    public static void Install(HarmonyInstance harmony, Func<PlaneFlight, bool> stillAuthorized)
    {
        harmony.Patch(PilotInputMethod, prefix: Hook(nameof(BeforePilotInput)), finalizer: Hook(nameof(AfterPilotInput)));
        harmony.Patch(Getters[0], postfix: Hook(nameof(Pitch)));
        harmony.Patch(Getters[1], postfix: Hook(nameof(Roll)));
        harmony.Patch(Getters[2], postfix: Hook(nameof(Yaw)));
        harmony.Patch(Getters[4], postfix: Hook(nameof(Brakes)));
        _stillAuthorized = stillAuthorized;
    }

    public static bool Publish(PlaneFlight plane, double until, LandingAxes axes, float[] action)
    {
        if (_stillAuthorized?.Invoke(plane) != true || _manualRelease
            || Time.unscaledTimeAsDouble >= until)
        {
            Release();
            return false;
        }

        _lease = new Lease(plane.Pointer, until, axes, action);
        return true;
    }

    public static bool ConsumeManualRelease()
    {
        bool released = _manualRelease;
        _manualRelease = false;
        return released;
    }

    public static bool ConsumeAuthorityLoss()
    {
        bool lost = _authorityLost;
        _authorityLost = false;
        return lost;
    }

    public static void Release() => _lease = null;

    public static void Uninstall(HarmonyInstance harmony)
    {
        Release();
        _stillAuthorized = null;
        _manualRelease = false;
        _authorityLost = false;
        harmony.UnpatchSelf();
    }

    private static MethodInfo Getter(string name) => AccessTools.PropertyGetter(typeof(GameInput), name)
        ?? throw new MissingMethodException(typeof(GameInput).FullName, name);

    private static HarmonyMethod Hook(string method) => new(typeof(LandingInput), method);

    private static void BeforePilotInput(PlaneFlight __instance, out bool __state)
    {
        __state = _readingForFlight;
        _readingForFlight = false;
        var lease = _lease;
        if (lease == null) return;
        try
        {
            if (Time.unscaledTimeAsDouble >= lease.Until || __instance.Pointer != lease.Plane
                || _stillAuthorized?.Invoke(__instance) != true)
            {
                _authorityLost = true;
                Release();
                return;
            }

            // Check the complete unmodified input before any axis in this packet can be substituted.
            if (PilotSample.Read().Takeover || __instance.LocalHandOnThrottle)
            {
                _manualRelease = true;
                Release();
                return;
            }
            _readingForFlight = true;
        }
        catch (Exception)
        {
            _authorityLost = true;
            Release();
        }
    }

    private static void AfterPilotInput(bool __state) => _readingForFlight = __state;

    private static void Pitch(ref float __result) => Substitute(ref __result, LandingAxes.Pitch, 0);
    private static void Roll(ref float __result) => Substitute(ref __result, LandingAxes.Roll, 1);
    private static void Yaw(ref float __result) => Substitute(ref __result, LandingAxes.Yaw, 2);

    private static void Substitute(ref float original, LandingAxes axis, int index)
    {
        var lease = _lease;
        if (_readingForFlight && lease != null && (lease.Axes & axis) != 0) original = lease.Action[index];
    }

    private static void Brakes(ref bool __result)
    {
        var lease = _lease;
        if (_readingForFlight && lease != null && (lease.Axes & LandingAxes.Brake) != 0)
            __result = lease.Action[4] >= 0.5f;
    }
}
#endif
