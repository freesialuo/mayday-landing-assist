// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using System.Reflection;
using MelonLoader;
#if LANDING_AUTO
using HarmonyLib;
using HarmonyInstance = HarmonyLib.Harmony;
#endif

namespace MaydayLanding;

internal sealed class ControllerConflict
{
    private const BindingFlags Fields = BindingFlags.Public | BindingFlags.NonPublic
        | BindingFlags.Instance | BindingFlags.Static;
    private const string LegacyAssembly = "MaydayAutoFlight";
    private const string LegacyPatchType = "MaydayAutoFlight.Game.InputPatch";
    private const string AutoAssembly = "MaydayLandingAuto";
    private readonly Dictionary<Type, LegacyState> _legacy = new();

    public static bool AutoInstalled(MelonMod own)
    {
        foreach (var melon in MelonBase.RegisteredMelons)
            if (!ReferenceEquals(melon, own) && IsLandingAuto(melon.GetType())) return true;
        return false;
    }

    private static bool IsLandingAuto(Type type) =>
        type.FullName == typeof(LandingMod).FullName && type.Assembly.GetName().Name == AutoAssembly;

    public bool HasController(MelonMod own)
    {
        foreach (var melon in MelonBase.RegisteredMelons)
        {
            if (ReferenceEquals(melon, own)) continue;
            Type type = melon.GetType();
            if (type.Assembly.GetName().Name == LegacyAssembly)
            {
                if (!_legacy.TryGetValue(type, out var state))
                {
                    state = new LegacyState(type);
                    _legacy.Add(type, state);
                }
                if (state.Active(melon)) return true;
                continue;
            }

            if (IsLandingAuto(type)) return true;
        }
        return false;
    }

#if LANDING_AUTO
    public static bool HasUnknownPatch(HarmonyInstance own)
    {
        foreach (var getter in LandingInput.Getters)
            if (HasUnknownPatch(own, getter)) return true;
        return HasUnknownPatch(own, LandingInput.PilotInputMethod);
    }

    private static bool HasUnknownPatch(HarmonyInstance own, MethodInfo method)
    {
        var patches = HarmonyInstance.GetPatchInfo(method);
        if (patches == null) return false;
        return Unknown(patches.Prefixes) || Unknown(patches.Postfixes) || Unknown(patches.Transpilers)
            || Unknown(patches.Finalizers);

        bool Unknown(IEnumerable<Patch> items)
        {
            foreach (var patch in items)
            {
                if (patch.owner == own.Id) continue;
                var type = patch.PatchMethod.DeclaringType;
                if (type?.FullName == LegacyPatchType && type.Assembly.GetName().Name == LegacyAssembly) continue;
                return true;
            }
            return false;
        }
    }
#endif

    private sealed class LegacyState
    {
        private readonly FieldInfo? _controller;
        private readonly PropertyInfo? _master;
        private readonly FieldInfo? _sticks;
        private readonly FieldInfo? _brakes;

        public LegacyState(Type modType)
        {
            _controller = modType.GetField("_af", Fields);
            _master = _controller?.FieldType.GetProperty("Master", Fields);
            var patch = modType.Assembly.GetType(LegacyPatchType);
            _sticks = patch?.GetField("Inject", Fields);
            _brakes = patch?.GetField("BrakeInject", Fields);
        }

        public bool Active(MelonBase mod)
        {
            try
            {
                var controller = _controller?.GetValue(mod);
                return controller == null || _master?.GetValue(controller)?.ToString() != "Off"
                    || _sticks?.GetValue(null) is not false || _brakes?.GetValue(null) is not false;
            }
            catch (Exception)
            {
                return true;
            }
        }
    }
}
