// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;

namespace MaydayLanding.Tests;

internal static class DirectorTests
{
    private static readonly LandingIntent Ready = new(PolicyValidated: true);
    private static readonly LandingIntent Arm = Ready with { Engage = true };

    public static void Run()
    {
        AdviceWithoutPolicy();
        HandFlownRepairGuidance();
        GuidanceStandsByForCruiseAndDeparture();
        ManualRolloutRequiresArrivalEvidence();
        LeavingApproachAreaReleases();
        PilotOverridesLatchOff();
        StaleOrDiscontinuousMotionReleases();
        VerifiedAndUnverifiedCrew();
        JammedGearRefusesArmAndAdvisesManual();
        JammedGearReleasesControl();
        ArmingOutsideTheWindowEngagesLater();
        LockedGearWithJamFlagCanOperate();
        ReverseOnlyDuringFastNormalRollout();
        RepairAndLastingSingleEngine();
        InadequatePowerAndLateGoAround();
        CircuitReentersOnce();
        CircuitNeedsNewRings();
        CircuitTimeoutReleases();
        RemainBrakedUntilStoppedAndSecured();
    }

    private static void AdviceWithoutPolicy()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame["cross"] = 55f;
        frame["turbulence"] = 0.9f;
        var result = director.Step(frame, new LandingIntent(Engage: true));
        IsManual(result, "Unaccepted candidate");
        Contains(result, "No accepted policy");
        Contains(result, "55 m right");
        Contains(result, "Heavy turbulence");

        frame.Seated = false;
        var observer = director.Step(frame, Ready);
        IsManual(observer, "Observer");
        Contains(observer, "Observer mode");
        Contains(observer, "m/s");
    }

    private static void HandFlownRepairGuidance()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame.PilotOverride = true;
        frame["cross"] = -40f;
        FailLeft(frame, 0.42f);
        var handFlown = director.Step(frame);
        IsManual(handFlown, "Hand-flown repair approach never actuates");
        Contains(handFlown, "Left repair 42%");
        Contains(handFlown, "40 m left");
        Contains(handFlown, "m/s");
        Check(handFlown.GuidanceActive && handFlown.LandingConfigurationGuidance, "Hand-flying in the approach keeps guidance active");

        frame.ExternalController = true;
        var legacy = Step(director, frame, default);
        Contains(legacy, "Another controller has control");
        Contains(legacy, "40 m left");
        frame.ExternalController = false;
        frame.NativeAp = true;
        frame["along"] = -16000f;
        frame["agl"] = 2000f;
        var cruise = Step(director, frame, default);
        Contains(cruise, "Native autopilot has control");
        Contains(cruise, "Outside approach area");
        Contains(cruise, "Left repair 42%");
        Check(!cruise.Status.Contains("m/s"), "Cruise should not display a final approach speed cue");
    }

    private static void GuidanceStandsByForCruiseAndDeparture()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame["along"] = -900f;
        frame["agl"] = 52f;
        frame["gear"] = 0f;
        frame["flap"] = 1f;
        frame["ice"] = 0.35f;
        FailLeft(frame, 0.42f);
        var otherLeg = Ready with { RouteStage = LandingRouteStage.Other };
        var departure = director.Step(frame, otherLeg);
        IsManual(departure, "Departure is monitoring only even with approach-like geometry");
        Check(!departure.GuidanceActive && !departure.LandingConfigurationGuidance, "Departure suppresses model and configuration guidance");
        Contains(departure, "Wing ice");
        Contains(departure, "Left repair 42%");
        Check(!departure.Status.Contains("Check landing gear", StringComparison.Ordinal) &&
            !departure.Status.Contains("m/s", StringComparison.Ordinal), "Departure suppresses gear and speed cues");

        var arrivalLeg = Ready with { RouteStage = LandingRouteStage.Approach };
        var arrival = Step(director, frame, arrivalLeg);
        Check(arrival.GuidanceActive && arrival.LandingConfigurationGuidance, "Same data on an arrival keeps local approach guidance");
        Contains(arrival, "Check landing gear");

        frame["along"] = -12000f;
        frame["agl"] = 1000f;
        var high = Step(director, frame, arrivalLeg);
        Check(!high.GuidanceActive, "An arrival leg outside runway geometry stays on standby");
        Contains(high, "Outside approach area");

        frame["along"] = -800f;
        frame["agl"] = 60f;
        frame["sink"] = -5f;
        var ambiguousClimb = Step(director, frame, Ready);
        Check(!ambiguousClimb.GuidanceActive, "Without route data a steep climb is not an approach");

        var plausible = Approach();
        var explicitDeparture = new LandingDirector().Step(plausible, Arm with { RouteStage = LandingRouteStage.Other });
        IsManual(explicitDeparture, "Do not arm approach control during a known non-arrival leg despite plausible geometry");
    }

    private static void ManualRolloutRequiresArrivalEvidence()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame["along"] = 120f;
        frame["agl"] = 0f;
        frame["on_ground"] = 1f;
        frame["ground_speed"] = 22f;
        var parkedOrTakingOff = director.Step(frame, Ready);
        Check(!parkedOrTakingOff.GuidanceActive, "Runway presence alone does not identify a landing");
        Check(!parkedOrTakingOff.Status.Contains("slow down", StringComparison.Ordinal), "No brake cue for takeoff");

        var arrival = Ready with { RouteStage = LandingRouteStage.Rollout };
        var nativeRollout = Step(director, frame, arrival);
        IsManual(nativeRollout, "Native landing rollout stays advisory");
        Check(nativeRollout.GuidanceActive && nativeRollout.Phase == LandingPhase.Rollout && !nativeRollout.LandingConfigurationGuidance,
            "Native rollout is sufficient when native ring telemetry has cleared");
        Contains(nativeRollout, "Hold the runway centerline and slow down");

        frame["cross"] = 55f;
        Check(!Step(director, frame, arrival).GuidanceActive, "Destination rollout off the paved runway is not a valid control context");
        frame["cross"] = 0f;
        frame.GateMask = 1;
        Check(Step(director, frame, Ready).GuidanceActive, "Actual ring verdicts cover rollout before the native stage updates");
        Check(!Step(director, frame, Ready with { RouteStage = LandingRouteStage.Other }).GuidanceActive,
            "An explicit departure or taxi stage takes precedence over old ring verdicts");

        frame["ground_speed"] = 0.05f;
        Check(!Step(director, frame, arrival).GuidanceActive, "Manual parked aircraft no longer requests braking");
    }

    private static void LeavingApproachAreaReleases()
    {
        var director = new LandingDirector();
        var frame = Approach();
        Check(director.Step(frame, Arm).Engaged, "Approach control starts inside the destination geometry");
        var outside = Advance(director, frame, frame["along"], 2030f, frame["agl"]);
        IsManual(outside, "Approach control releases if the aircraft leaves the destination geometry");
        Check(outside.Released && !outside.GuidanceActive, "Leaving the approach region ends proposals in the release frame");
    }

    private static void PilotOverridesLatchOff()
    {
        var director = new LandingDirector();
        var frame = Approach();
        Check(director.Step(frame, Arm).Engaged, "Fresh explicit arm should engage");
        frame.PilotOverride = true;
        var released = Step(director, frame, Arm);
        Check(released.Released, "Pilot override should emit release");
        IsManual(released, "Pilot override");

        frame.PilotOverride = false;
        IsManual(Step(director, frame, Arm), "Held engagement request after override");
        IsManual(Step(director, frame, Ready), "Waiting for a new engagement request");
        Check(Step(director, frame, Arm).Engaged, "New explicit arm should engage");

        frame.NativeAp = true;
        IsManual(Step(director, frame, Ready), "Native AP active");
        frame.NativeAp = false;
        IsManual(Step(director, frame, Ready), "Native AP stopped without an explicit arm");
        frame.ExternalController = true;
        IsManual(Step(director, frame, Arm), "Other controller active");
    }

    private static void StaleOrDiscontinuousMotionReleases()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame["observation_age"] = 0.3f;
        Check(director.Step(frame, Arm).Engaged, "0.3 sec should be within the actuation envelope");
        frame["observation_age"] = 0.31f;
        var late = Step(director, frame, Ready);
        IsManual(late, "0.31 sec actuation attempt");
        Check(late.Released, "Observed delay crossing the limit should release");
        Contains(late, "latency exceeded");

        frame["observation_age"] = 0.01f;
        Check(Step(director, frame, Arm).Engaged, "Explicit re-arm with fresh data");
        frame.SessionId += 1;
        IsManual(Step(director, frame, Ready), "Seat or airport identity changed");
        Check(Step(director, frame, Arm).Engaged, "Re-arm for the new identity");
        frame["along"] += 300f;
        IsManual(Step(director, frame, Ready), "Discontinuous position sample");
        Check(Step(director, frame, Arm).Engaged, "Re-arm after discontinuity");
        frame.Time -= 2;
        IsManual(director.Step(frame, Ready), "Clock rewind");
        frame.MotionSourceTime = 100d;
        Check(Step(director, frame, Arm).Engaged, "Valid snapshot clock after explicit re-arm");
        frame.MotionSourceTime = 99d;
        IsManual(Step(director, frame, Ready), "Native snapshot clock rewind with stable local clock");
        frame["observation_age"] = float.NaN;
        IsManual(Step(director, frame, Arm), "Unknown observation age");
    }

    private static void VerifiedAndUnverifiedCrew()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame.Host = false;
        frame["other_known"] = 0f;
        var blocked = director.Step(frame, Arm);
        IsManual(blocked, "Unmonitored remote crew");
        Contains(blocked, "Copilot controls unverified");
        frame.RemoteResponseMonitored = true;
        IsManual(Step(director, frame, Ready), "Monitored but no explicit arm");
        Check(Step(director, frame, Arm).Engaged, "Monitored remote crew with valid authorization");
        frame.OtherPilotOverride = true;
        IsManual(Step(director, frame, Ready), "Detected remote pilot override");
    }

    private static void JammedGearRefusesArmAndAdvisesManual()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame.GearJammed = true;
        frame["gear"] = 0.89f;

        var advice = director.Step(frame);
        IsManual(advice, "An Advisor without a policy warns about the jam");
        Contains(advice, "Use the hand crank before landing");
        Check(!advice.GuidanceActive, "Jammed approach does not offer model stick or throttle recommendations");

        var engaged = Step(director, frame, Arm);
        Check(engaged.Engaged, "A jam no longer refuses engagement; the crew can crank while the assist flies");
        Contains(engaged, "Use the hand crank before landing");

        var off = Step(director, frame, Ready with { Stop = true });
        frame.NativeAp = true;
        var autopilot = Step(director, frame, Ready);
        Check(autopilot.Status.StartsWith("Native autopilot", StringComparison.Ordinal), "An AP status takes precedence over a jam");
        frame.NativeAp = false;
        frame.Seated = false;
        var observer = Step(director, frame, Arm);
        Check(observer.Status.StartsWith("Observer mode", StringComparison.Ordinal), "Seat authorization takes precedence over a jam");
        IsManual(observer, "Observer cannot engage while gear is jammed");
    }

    private static void JammedGearReleasesControl()
    {
        // The assist never moves the gear: below 65 m AGL an unlocked gear makes the approach
        // unstable, so it goes around once and leaves the gear problem to the crew.
        var director = new LandingDirector();
        var frame = Approach();
        Check(director.Step(frame, Arm).Engaged, "Valid approach engages before a gear jam");
        frame.GearJammed = true;
        frame["gear"] = 0.6f;
        var continuing = Step(director, frame, Ready);
        Check(continuing.Engaged, "An airborne jam above the decision height keeps the assist flying");
        Contains(continuing, "Use the hand crank before landing");
        var low = Advance(director, frame, -1300f, 0f, 60f);
        Check(low.Engaged && low.Phase == LandingPhase.GoAround, "Unlocked gear below 65 m goes around");

        frame["gear"] = 0.89f;
        frame.PilotOverride = true;
        var pilot = Step(director, frame, Ready);
        Check(pilot.Released && pilot.Status == "Pilot has control.", "Immediate pilot takeover takes priority over a gear warning");
    }

    private static void ArmingOutsideTheWindowEngagesLater()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame["along"] = -12000f;
        frame["agl"] = 650f;
        var armed = Step(director, frame, Arm);
        Check(!armed.Engaged && armed.Armed, "F7 outside the capture window arms");
        Contains(armed, "Armed.");
        Contains(armed, "distance:");
        frame.PilotOverride = true;
        frame["along"] = -7000f;
        frame["agl"] = 380f;
        var held = Step(director, frame, Ready);
        Check(!held.Engaged && held.Armed, "Pilot input keeps the arm without fighting the pilot");
        Contains(held, "Release the controls");
        frame.PilotOverride = false;
        frame.NativeAp = true;
        var ap = Step(director, frame, Ready);
        Check(ap.DisconnectAp && !ap.Engaged, "Inside the window with the native autopilot on: request its disconnect");
        frame.NativeAp = false;
        var engaged = Step(director, frame, Ready);
        Check(engaged.Engaged && !engaged.Armed, "Once the autopilot is off the armed assist engages");

        var other = new LandingDirector();
        var far = Approach();
        far["along"] = -12000f;
        far["agl"] = 650f;
        Check(Step(other, far, Arm).Armed, "Arms");
        Check(Step(other, far, Ready).Armed, "Releasing F7 keeps the arm");
        var cancelled = Step(other, far, Arm);
        Check(!cancelled.Armed && !cancelled.Engaged, "A second F7 cancels the arm");
    }

    private static void ReverseOnlyDuringFastNormalRollout()
    {
        var director = new LandingDirector();
        var frame = Approach();
        Check(director.Step(frame, Arm).Engaged, "Valid approach engages");
        Advance(director, frame, 80f, 0f, 8f);
        frame["on_ground"] = 1f;
        frame["agl"] = 0f;
        frame["ground_speed"] = 35f;
        var fast = Step(director, frame, Ready);
        Check(fast.Engaged && fast.Phase == LandingPhase.Rollout && fast.UseReverse, "Fast normal rollout asks for reverse");
        Contains(fast, "Reverse thrust");
        frame["ground_speed"] = 9f;
        var slow = Step(director, frame, Ready);
        Check(slow.Engaged && !slow.UseReverse, "Reverse is stowed below the off speed");
        frame["ground_speed"] = 35f;
        frame["right_state"] = 0f;
        var single = Step(director, frame, Ready);
        Check(!single.UseReverse, "No reverse without both engines");
    }

    private static void LockedGearWithJamFlagCanOperate()
    {
        var director = new LandingDirector();
        var frame = Approach();
        frame.GearJammed = true;
        frame["gear"] = 0.9f;
        Check(director.Step(frame, Arm).Engaged, "A jam flag does not block gear at the lock threshold");
        var flare = Advance(director, frame, 80f, 0f, 8f);
        Check(flare.Engaged && flare.Phase == LandingPhase.Flare, "Locked gear retains the flare controller");

        frame["on_ground"] = 1f;
        frame["agl"] = 0f;
        frame["gear"] = 0.8f;
        var rollout = Step(director, frame, Ready);
        Check(rollout.Engaged && rollout.Phase == LandingPhase.Rollout && (rollout.Axes & LandingAxes.Brake) != 0,
            "Airborne gear fault protection does not cancel on-runway braking");
    }

    private static void RepairAndLastingSingleEngine()
    {
        var healthyOnly = new LandingDirector();
        var restricted = Approach();
        FailLeft(restricted, repair: 0.5f);
        IsManual(healthyOnly.Step(restricted, Arm), "Healthy-only accepted policy during failure");

        var director = new LandingDirector();
        var frame = Approach();
        var degraded = Ready with { SupportsDegraded = true };
        Check(director.Step(frame, degraded with { Engage = true }).Engaged, "Arm for validated engine degradation");
        FailLeft(frame, repair: 0.35f);
        var repair = Step(director, frame, degraded);
        Check(repair.Engaged, "One usable engine should preserve control while repair progresses");
        Contains(repair, "Left repair 35%");

        frame["left_repair"] = 0.75f;
        Contains(Step(director, frame, degraded), "Left repair 75%");
        frame["left_state"] = 2f;
        frame["left_thrust"] = 0.65f;
        frame["left_health"] = 0.9f;
        frame["left_repair"] = 1f;
        var restored = Step(director, frame, degraded);
        Check(restored.Engaged && !restored.Status.Contains("Single-engine"), "Restored engine should return to normal guidance");

        FailLeft(frame, repair: 0f);
        for (int i = 0; i < 15; i++)
        {
            var lastingSingle = Step(director, frame, degraded);
            Check(lastingSingle.Engaged, "Stable single-engine approach should continue without a timeout");
            Contains(lastingSingle, "Single-engine power");
        }
        frame["right_thrust"] = 0.01f;
        IsManual(Step(director, frame, degraded), "Measured thrust loss on remaining engine");

        var apparentPower = new LandingDirector();
        var noThrust = Approach();
        noThrust["left_thrust"] = noThrust["right_thrust"] = 0f;
        Check(apparentPower.Step(noThrust, degraded with { Engage = true }).Engaged, "Coasting without demand is allowed");
        noThrust["throttle"] = 1f;
        Check(Step(apparentPower, noThrust, degraded).Engaged, "Allow a brief engine spool delay");
        Step(apparentPower, noThrust, degraded);
        var insufficient = Step(apparentPower, noThrust, degraded);
        IsManual(insufficient, "No measured thrust with sustained demand despite running flags");
        Contains(insufficient, "forced landing");
    }

    private static void InadequatePowerAndLateGoAround()
    {
        var director = new LandingDirector();
        var frame = Approach();
        var degraded = Ready with { SupportsDegraded = true };
        Check(director.Step(frame, degraded with { Engage = true }).Engaged, "Arm before observed power loss");
        FailLeft(frame, 0f);
        frame["right_thrust"] = 0.3f;
        Check(Step(director, frame, degraded).Engaged, "Adequate approach margin with insufficient climb margin");
        var result = Step(director, frame, degraded with { GoAround = true });
        IsManual(result, "Insufficient climb margin for automatic go-around");
        Contains(result, "climb unproven");

        var noPower = new LandingDirector();
        var unusable = Approach();
        Check(noPower.Step(unusable, degraded with { Engage = true }).Engaged, "Arm before dual failure");
        FailLeft(unusable, 0f);
        unusable["right_state"] = 3f;
        Contains(Step(noPower, unusable, degraded), "forced landing");

        var low = new LandingDirector();
        var landing = Approach();
        low.Step(landing, Arm);
        Advance(low, landing, -400f, 0f, 22f);
        var below = Step(low, landing, Ready with { GoAround = true });
        IsManual(below, "Late go-around outside envelope");
        Contains(below, "outside its envelope");
    }

    private static void CircuitReentersOnce()
    {
        var director = new LandingDirector();
        var frame = Approach();
        director.Step(frame, Arm);
        frame.GateMissed = true;
        frame.GateMask = 2;
        frame["next_gate_index"] = 1;
        CheckTarget(Step(director, frame, Ready), 1750f, 0f, "Missed ring starts runway-aligned climb");
        double started = frame.Time;
        var activeCircuit = Step(director, frame, Ready with { GoAround = true, RouteStage = LandingRouteStage.Other });
        CheckTarget(activeCircuit, 1750f, 0f, "Repeated go-around cannot restart the circuit");
        Check(activeCircuit.GuidanceActive && !activeCircuit.LandingConfigurationGuidance,
            "An engaged outbound circuit keeps policy inference without approach configuration cues");
        CheckTarget(Advance(director, frame, 1470f, 0f, 120f), 1750f, 1500f, "Crosswind starts after climbing");
        CheckTarget(Advance(director, frame, 1700f, 1220f, 150f), -4650f, 1500f, "Downwind waypoint");
        CheckTarget(Advance(director, frame, -4320f, 1500f, 180f), -4850f, 0f, "Base leg waypoint");
        CheckTarget(Advance(director, frame, -4650f, 440f, 200f), -3500f, 0f, "Reorient toward final");

        // The historical missed flag remains visible, while the native ring set resets.
        frame.GateMask = 0;
        frame["next_gate_index"] = 0;
        var retry = Advance(director, frame, -4450f, 0f, 230f);
        Check(retry.Engaged && retry.Reentries == 1 && retry.Phase != LandingPhase.GoAround, "A fresh circuit may reenter once");
        Check(frame.Time - started > 240 && frame.Time - started < 300, "A long but valid circuit exceeds the old 240s deadline");
        frame.GateMask = 2;
        frame["next_gate_index"] = 1;
        var secondMiss = Step(director, frame, Ready);
        IsManual(secondMiss, "Second missed ring should release");
        Contains(secondMiss, "Retry used");
    }

    private static void CircuitNeedsNewRings()
    {
        var director = new LandingDirector();
        var frame = Approach();
        director.Step(frame, Arm);
        frame.GateMissed = true;
        frame.GateMask = 2;
        frame["next_gate_index"] = 1;
        Step(director, frame, Ready);
        Advance(director, frame, 1470f, 0f, 120f);
        Advance(director, frame, 1700f, 1220f, 150f);
        Advance(director, frame, -4320f, 1500f, 180f);
        Advance(director, frame, -4650f, 440f, 200f);
        var spent = Advance(director, frame, -4450f, 0f, 230f);
        IsManual(spent, "No retry on a stale ring set");
        Check(spent.Reentries == 0, "Stale ring data must not consume an approved retry");
        Contains(spent, "rings did not reset");
    }

    private static void CircuitTimeoutReleases()
    {
        var director = new LandingDirector();
        var frame = Approach();
        director.Step(frame, Arm);
        var result = Step(director, frame, Ready with { GoAround = true });
        CheckTarget(result, 1750f, 0f, "Go-around starts");
        for (int i = 0; i < 600; i++)
            result = Step(director, frame, Ready);
        Check(result.Engaged, "Circuit remains active through 300s");
        result = Step(director, frame, Ready);
        IsManual(result, "Circuit releases after the 300s deadline");
        Contains(result, "outside its envelope");
    }

    private static void RemainBrakedUntilStoppedAndSecured()
    {
        var director = new LandingDirector();
        var frame = Approach();
        director.Step(frame, Arm);
        var flare = Advance(director, frame, 80f, 0f, 8f);
        Check(flare.Phase == LandingPhase.Flare && (flare.Axes & LandingAxes.Brake) == 0, "Airborne flare owns flight axes only");
        frame["on_ground"] = 1f;
        frame["agl"] = 0f;
        var rollout = Step(director, frame, Ready);
        Check(rollout.Phase == LandingPhase.Rollout && (rollout.Axes & LandingAxes.Brake) != 0, "On-runway rollout owns brake");
        frame["on_ground"] = 0f;
        frame["agl"] = 2f;
        var bounce = Step(director, frame, Ready);
        Check(bounce.Phase == LandingPhase.Flare && bounce.Engaged && (bounce.Axes & LandingAxes.Brake) == 0,
            "A recoverable bounce uses flare instead of ground brakes");
        frame["on_ground"] = 1f;
        frame["agl"] = 0f;
        frame["ground_speed"] = 0.09f;
        var hold = Step(director, frame, Ready);
        Check(hold.HoldBrake && hold.Axes == (LandingAxes.Brake | LandingAxes.Throttle), "Brake and idle only near standstill");
        var secured = Ready with { ParkingBrakeSet = true };
        for (int i = 0; i < 5; i++)
            Check(Step(director, frame, secured).Engaged, "Parking brake already set still requires a sustained stop");

        frame["ground_speed"] = 0.1f;
        Check(Step(director, frame, secured).Engaged, "At the threshold the stop timer resets");
        frame["ground_speed"] = 0.08f;
        for (int i = 0; i < 7; i++)
            hold = Step(director, frame, Ready);
        Check(hold.Engaged && hold.HoldBrake && hold.SecureParkingBrake, "Keep braking at a full stop without parking confirmation");
        Contains(hold, "Securing parking brake");
        var stopped = Step(director, frame, secured);
        IsManual(stopped, "Complete stop confirmed and secured");
        Check(stopped.Released && stopped.Phase == LandingPhase.Rollout, "Complete rollout emits release and keeps HUD phase");
    }

    private static LandingFrame Approach()
    {
        var frame = new LandingFrame { Time = 1, SessionId = 1, Valid = true, Seated = true, Host = true };
        frame["along"] = -3200f;
        frame["agl"] = 200f;
        frame["speed"] = frame["ground_speed"] = 48f;
        frame["sink"] = 0f;
        frame["track_cos"] = frame["heading_cos"] = 1f;
        frame["gear"] = 1f;
        frame["flap"] = 2f;
        frame["other_known"] = 1f;
        frame["runway_length"] = 1750f;
        frame["runway_half_width"] = 28f;
        frame["next_gate_distance"] = 500f;
        frame["left_valid"] = frame["right_valid"] = 1f;
        frame["left_state"] = frame["right_state"] = 2f;
        frame["left_health"] = frame["right_health"] = 1f;
        frame["left_thrust"] = frame["right_thrust"] = 0.68f;
        return frame;
    }

    private static void FailLeft(LandingFrame frame, float repair)
    {
        frame["left_state"] = 3f;
        frame["left_thrust"] = 0f;
        frame["left_repair"] = repair;
    }

    private static LandingDecision Step(LandingDirector director, LandingFrame frame, LandingIntent intent)
    {
        frame.Time += 0.5;
        return director.Step(frame, intent);
    }

    private static LandingDecision Advance(LandingDirector director, LandingFrame frame, float along, float cross, float agl)
    {
        float startAlong = frame["along"], startCross = frame["cross"], startAgl = frame["agl"];
        int steps = Math.Max(1, (int)MathF.Ceiling(MathF.Max(MathF.Abs(along - startAlong), MathF.Abs(cross - startCross)) / 27f));
        LandingDecision decision = default;
        for (int i = 1; i <= steps; i++)
        {
            float fraction = (float)i / steps;
            frame["along"] = startAlong + (along - startAlong) * fraction;
            frame["cross"] = startCross + (cross - startCross) * fraction;
            frame["agl"] = startAgl + (agl - startAgl) * fraction;
            decision = Step(director, frame, Ready);
            if (decision.Released) break;
        }
        return decision;
    }

    private static void IsManual(LandingDecision result, string scenario) =>
        Check(!result.Engaged && result.Axes == LandingAxes.None && !result.HoldBrake, scenario);

    private static void CheckTarget(LandingDecision result, float along, float cross, string scenario) =>
        Check(result.Engaged && result.Phase == LandingPhase.GoAround && result.TargetAlong == along && result.TargetCross == cross,
            $"{scenario} ({result.Status})");

    private static void Contains(LandingDecision result, string expected) =>
        Check(result.Status.Contains(expected, StringComparison.Ordinal), $"Expected '{expected}' in '{result.Status}'");

    private static void Check(bool condition, string description)
    {
        if (!condition) throw new InvalidOperationException(description);
    }
}
