using System;

namespace MaydayLanding;

/// <summary>Visible-state guidance for the bounded-residual landing candidate.</summary>
public static class LandingPrior
{
    // Keep the physics constants and ordering in sync with mayday_landing/observable_guidance.py.
    private const float AimOffset = 250f;
    private const float FirstRing = 4000f;
    private const float GlideSlope = 0.05240777928304121f;
    private const float Rad = 0.01745329f;
    private const float Gravity = 9.81f;

    public static float[] Predict(float[] observation, LandingPhase phase, ReadOnlySpan<float> stallSpeeds)
    {
        if (observation.Length != LandingFeatures.Names.Length)
            throw new ArgumentException("Observation length does not match the landing feature schema.", nameof(observation));
        if (stallSpeeds.Length != 3)
            throw new ArgumentException("Three flap-notch stall speeds are required.", nameof(stallSpeeds));

        float Feature(string name) => observation[LandingFeatures.Index(name)];

        float age = Math.Clamp(Feature("observation_age"), 0f, 0.75f);
        float groundSpeed = Feature("ground_speed");
        float measuredSpeed = Feature("speed");
        float speed = MathF.Max(measuredSpeed, 15f);
        float sink = Feature("sink");
        float velocityAlong = groundSpeed * Feature("track_cos");
        float velocityCross = groundSpeed * Feature("track_sin");
        float track = MathF.Atan2(velocityCross, velocityAlong) / Rad;
        float heading = MathF.Atan2(Feature("heading_sin"), Feature("heading_cos")) / Rad;
        float airTrack = MathF.Atan2(velocityCross - Feature("wind_cross"), velocityAlong - Feature("wind_along")) / Rad;
        float beta = Wrap180(heading - airTrack);

        float along = Feature("along") - AimOffset + age * velocityAlong;
        float cross = Feature("cross") + age * velocityCross;
        float height = MathF.Max(0f, Feature("agl") - age * sink);
        float gear = Feature("gear");
        float flap = Feature("flap");
        float bank = Feature("roll");
        float yawRate = Feature("yaw_rate");
        float predictedHeading = heading + MathF.Min(0.3f, age) * yawRate;
        bool onGround = Feature("on_ground") > 0.5f || phase == LandingPhase.Rollout;
        bool circuit = phase == LandingPhase.GoAround;
        bool flare = !circuit && (phase == LandingPhase.Flare || height <= 10.5f);

        float toRing = MathF.Max(-FirstRing - along, 0f);
        float lookAhead = along < -FirstRing ? Math.Clamp(0.47f * toRing, 190f, 450f) : flare ? 155f : 260f;
        float pathTrack = Math.Clamp(MathF.Atan2(-cross, lookAhead) / Rad, -48f, 48f);
        float waypointAlong = Feature("target_along") - AimOffset;
        float waypointCross = Feature("target_cross");
        float circuitTrack = MathF.Atan2(waypointCross - cross, waypointAlong - along) / Rad;
        float turn = Wrap180((circuit ? circuitTrack : pathTrack) - track);
        float bankLimit = circuit ? 30f : flare ? 6f : 22f;
        float targetBank = Math.Clamp(0.79f * turn - 0.75f * beta, -bankLimit, bankLimit);
        float roll = (0.062f / (1f + 1.6f * age)) *
            (targetBank - bank - (0.22f + 0.25f * age) * Feature("roll_rate"));

        float engines = MathF.Max(0.25f, EngineCapacity(Feature("left_state"), Feature("left_health"), Feature("left_fire")) +
            EngineCapacity(Feature("right_state"), Feature("right_health"), Feature("right_fire")));
        float glideHeight = MathF.Max(-along * GlideSlope, 0f);
        float glideVertical = Math.Clamp(-GlideSlope * MathF.Max(10f, velocityAlong) + 0.092f * (glideHeight - height), -4.9f, 2.8f);
        float flareVertical = -Math.Clamp(0.18f * height, 0.75f, 1.7f);
        float circuitHeight = waypointAlong < -3800f && MathF.Abs(waypointCross) < 500f ? MathF.Max(240f, glideHeight) : 235f;
        float circuitVertical = Math.Clamp(0.09f * (circuitHeight - height), -2.4f, engines > 1.5f ? 5.3f : 1.7f);
        // Speed priority (guidance v2): trade climb for airspeed with iced or derated engines.
        circuitVertical = MathF.Min(circuitVertical, MathF.Max(0.35f * (measuredSpeed - 44f), 0f));
        float desiredVertical = circuit ? circuitVertical : flare ? flareVertical : glideVertical;
        float gamma = MathF.Asin(Math.Clamp(-sink / speed, -0.65f, 0.65f)) / Rad;
        float desiredGamma = MathF.Asin(Math.Clamp(desiredVertical / speed, -0.25f, 0.25f)) / Rad;
        float bankCosine = MathF.Max(MathF.Cos(bank * Rad), 0.75f);
        float lift = Math.Clamp(MathF.Cos(gamma * Rad) + 0.085f * (desiredGamma - gamma), 0.44f, 1.62f) / bankCosine;
        float proximity = Math.Clamp(1f - height / 12.5f, 0f, 1f);
        float targetPitch = Math.Clamp(gamma + RequiredAlpha(speed, flap, proximity * proximity, stallSpeeds) * lift - 2f,
            -10f, flare ? 6.8f : 14f);
        float pitchScale = 1f - 0.55f * Math.Clamp((12f - height) / 8f, 0f, 1f);
        float pitch = (0.36f / (1f + 1.9f * age)) *
            (targetPitch - Feature("pitch") - (0.19f + 0.18f * age) * Feature("pitch_rate")) / pitchScale;
        pitch = MathF.Abs(pitch) >= 0.022f ? MathF.CopySign(MathF.Max(MathF.Abs(pitch), 0.055f), pitch) : 0f;

        float yaw = flare && height < 3f ? -0.070f * predictedHeading : -0.20f * beta;
        yaw -= 0.58f * (Feature("left_thrust") - Feature("right_thrust"));
        float groundYaw = -Math.Clamp(0.040f * (predictedHeading + MathF.Atan2(cross, 38f) / Rad) + 0.026f * yawRate, -0.78f, 0.78f);
        float drag = (gear * 0.15f + 1f + flap * 0.1f) * speed * 0.0002318f * speed + 950.6f / (speed * speed);
        float wantedSpeed = circuit ? 52f : flare ? 43f : 47f;
        float thrust = 0.95f * engines * (1f - 0.35f * Feature("ice"));
        float throttle = (drag + Gravity * MathF.Sin(desiredGamma * Rad)) / thrust + 0.08f * (wantedSpeed - measuredSpeed);
        if (flare && height < 5.5f) throttle = 0f;

        if (onGround) return new[] { -0.42f, 0f, groundYaw, 0f, 1f };
        return new[] { Math.Clamp(pitch, -1f, 1f), Math.Clamp(roll, -1f, 1f), Math.Clamp(yaw, -1f, 1f),
            Math.Clamp(throttle, 0f, 1f), 0f };
    }

    private static float EngineCapacity(float state, float health, float fire) =>
        state == 2f ? (0.45f + 0.55f * health) * (fire > 0f ? 0.7f : 1f) : 0f;

    private static float RequiredAlpha(float speed, float flap, float effect, ReadOnlySpan<float> stallSpeeds)
    {
        float position = Math.Clamp(flap, 0f, 2f);
        int index = Math.Min((int)position, 1);
        float stallSpeed = (stallSpeeds[index + 1] - stallSpeeds[index]) * (position - index) + stallSpeeds[index];
        float ratio = stallSpeed / speed;
        return (1f - effect * 0.12f) * ratio * 14f * ratio;
    }

    private static float Wrap180(float angle)
    {
        float wrapped = Math.Clamp(angle - MathF.Floor(angle / 360f) * 360f, 0f, 360f);
        return wrapped > 180f ? wrapped - 360f : wrapped;
    }
}
