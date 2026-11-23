using System;

namespace MaydayLanding.Tests;

internal static class LandingPriorTests
{
    public static void Run()
    {
        // Expected actions come from Python guidance with the recorder's flap-notch tuning.
        Check(LandingPhase.Capture, -12f, -7.5f,
            new[] { 0.213067532f, -0.888240874f, -0.126564890f, 0.0480949879f, 0f },
            ("along", -5510f), ("cross", 300f), ("agl", 295f), ("speed", 51f), ("ground_speed", 52f),
            ("sink", 2.4f), ("pitch", -1.2f), ("roll", 5f), ("pitch_rate", 1.1f), ("roll_rate", -4f), ("yaw_rate", 2.1f),
            ("observation_age", 0.3f), ("wind_along", 3.2f), ("wind_cross", -4f), ("gear", 0.7f), ("flap", 1f));

        Check(LandingPhase.Final, 14f, 9.5f,
            new[] { 0.342919946f, 0.106531799f, 0.257807493f, 0.625441253f, 0f },
            ("along", -820f), ("cross", -49f), ("agl", 58f), ("speed", 46f), ("ground_speed", 45f),
            ("sink", 3.1f), ("pitch", 0.7f), ("roll", -6f), ("pitch_rate", -2.1f), ("roll_rate", 1.9f), ("yaw_rate", -0.9f),
            ("observation_age", 0.1f), ("wind_along", -1.6f), ("wind_cross", 4.1f), ("ice", 0.16f),
            ("left_state", 0f), ("left_health", 0.55f), ("left_thrust", 0f), ("right_thrust", 0.78f), ("flap", 0f));

        Check(LandingPhase.Flare, 2.2f, -1.1f,
            new[] { -0.211280212f, -0.0804881379f, 0.0918700024f, 0f, 0f },
            ("along", 330f), ("cross", -2.4f), ("agl", 2.4f), ("speed", 42f), ("ground_speed", 42.5f),
            ("sink", 1.8f), ("pitch", 1.1f), ("roll", 1.6f), ("pitch_rate", 1.4f), ("roll_rate", -0.8f), ("yaw_rate", 0.7f),
            ("observation_age", 0.17f), ("wind_cross", 1.6f), ("left_thrust", 0.26f), ("right_thrust", 0.3f));

        Check(LandingPhase.GoAround, -175f, 178f,
            new[] { 0.609214365f, 0.799736679f, 1f, 1f, 0f },
            ("along", -4380f), ("cross", 1250f), ("agl", 206f), ("speed", 49f), ("ground_speed", 50f),
            ("sink", -0.7f), ("pitch", -0.2f), ("roll", 4.2f), ("pitch_rate", -0.8f), ("roll_rate", 1.4f), ("yaw_rate", -2.3f),
            ("observation_age", 0.6f), ("wind_along", -2.9f), ("wind_cross", 0.9f), ("ice", 0.11f),
            ("target_along", -4850f), ("left_fire", 0.48f), ("left_health", 0.64f), ("left_thrust", 0.37f), ("right_thrust", 0.63f));

        // Below 44 m/s the circuit climb target is capped at level flight (speed priority).
        Check(LandingPhase.GoAround, -175f, 178f,
            new[] { 0.772415102f, 0.799736679f, 1f, 1f, 0f },
            ("along", -4380f), ("cross", 1250f), ("agl", 206f), ("speed", 41f), ("ground_speed", 50f),
            ("sink", -0.7f), ("pitch", -0.2f), ("roll", 4.2f), ("pitch_rate", -0.8f), ("roll_rate", 1.4f), ("yaw_rate", -2.3f),
            ("observation_age", 0.6f), ("wind_along", -2.9f), ("wind_cross", 0.9f), ("ice", 0.11f),
            ("target_along", -4850f), ("left_fire", 0.48f), ("left_health", 0.64f), ("left_thrust", 0.37f), ("right_thrust", 0.63f));

        Check(LandingPhase.Rollout, 1.8f, -2.4f,
            new[] { -0.42f, 0f, -0.147773251f, 0f, 1f },
            ("along", 520f), ("cross", 3.6f), ("speed", 25f), ("ground_speed", 22f), ("pitch", -2f),
            ("roll", 0.3f), ("pitch_rate", 1.1f), ("roll_rate", -0.2f), ("yaw_rate", 0.56f),
            ("observation_age", 0.2f), ("wind_cross", 2.2f));
    }

    private static void Check(LandingPhase phase, float track, float heading, float[] expected,
        params (string Name, float Value)[] features)
    {
        const float rad = 0.01745329f;
        var frame = new LandingFrame();
        frame["gear"] = 1f;
        frame["flap"] = 2f;
        frame["left_state"] = frame["right_state"] = 2f;
        frame["left_health"] = frame["right_health"] = 1f;
        frame["left_thrust"] = frame["right_thrust"] = 0.5f;
        frame["track_sin"] = MathF.Sin(track * rad);
        frame["track_cos"] = MathF.Cos(track * rad);
        frame["heading_sin"] = MathF.Sin(heading * rad);
        frame["heading_cos"] = MathF.Cos(heading * rad);
        foreach (var (name, value) in features) frame[name] = value;

        var actual = LandingPrior.Predict(frame.Values, phase, new[] { 32f, 29f, 26f });
        for (int axis = 0; axis < expected.Length; axis++)
            if (!float.IsFinite(actual[axis]) || MathF.Abs(actual[axis] - expected[axis]) > 5e-5f)
                throw new InvalidOperationException($"Guidance parity: {phase} axis {axis}, expected {expected[axis]:R}, received {actual[axis]:R}.");
    }
}
