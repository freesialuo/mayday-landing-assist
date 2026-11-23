// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace MaydayLanding.Tests;

internal static class ModelTests
{
    // NumPy Policy predictions from parity_fixture() and parity_sequence().
    private static readonly float[][] MlpExpected =
    {
        new[] { -0.268725961f, -0.040940233f, -0.010450272f, 0.505015314f, 0.551797807f },
        new[] { -0.217972904f, -0.044555791f, 0.023153918f, 0.522709727f, 0.557995260f },
        new[] { -0.296988696f, 0.099717647f, 0.129477143f, 0.540005803f, 0.522786498f },
        new[] { -0.182526186f, -0.212341100f, 0.128344983f, 0.534809113f, 0.537396848f },
        new[] { -0.232773408f, -0.044197943f, 0.012654717f, 0.517377377f, 0.566900492f },
    };

    private static readonly float[][] GruExpected =
    {
        new[] { 0.009755749f, -0.283903867f, -0.197155312f, 0.541429818f, 0.524508774f },
        new[] { -0.170348883f, -0.118624799f, -0.203367978f, 0.541096210f, 0.484611511f },
        new[] { -0.336154372f, 0.118910111f, -0.058183029f, 0.534976542f, 0.524254858f },
        new[] { -0.154553980f, -0.044569019f, 0.083774894f, 0.504691839f, 0.548523784f },
        new[] { -0.106465146f, -0.182154730f, -0.227694407f, 0.528430045f, 0.525972307f },
    };

    private static readonly float[][] ResidualGruExpected =
    {
        new[] { 0.990975559f, -1.000000000f, 0.966341376f, 1.000000000f, 0f },
        new[] { 0.972965121f, -0.997793734f, 0.965595841f, 1.000000000f, 0f },
        new[] { 0.956384599f, -0.962163508f, 0.983018041f, 1.000000000f, 0f },
        new[] { 0.974544585f, -0.986685395f, 1.000000000f, 0.996126056f, 1f },
        new[] { 0.979353487f, -1.000000000f, 0.962676704f, 1.000000000f, 0f },
    };

    public static void Run()
    {
        Verify(false, MlpExpected, MlpExpected[4]);
        Verify(true, GruExpected, new[] { -0.13971351f, -0.20069604f, -0.18420723f, 0.51624495f, 0.52991730f });
        VerifyGuided();

        var malformed = Fixture(false);
        malformed["feature_names"] = Enumerable.Reverse(LandingFeatures.Names).ToArray();
        Rejects<InvalidDataException>(() => LandingPolicy.Load(JsonSerializer.Serialize(malformed)));
        malformed = Fixture(true);
        malformed["std"] = new float[LandingFeatures.Names.Length];
        Rejects<InvalidDataException>(() => LandingPolicy.Load(JsonSerializer.Serialize(malformed)));
        var policy = LandingPolicy.Load(JsonSerializer.Serialize(Fixture(true)));
        Rejects<ArgumentException>(() => policy.Predict(new[] { 0f }, 0));
        Rejects<ArgumentOutOfRangeException>(() => policy.Predict(Observation(0), 5));
    }

    private static void VerifyGuided()
    {
        var speeds = new[] { 32f, 29f, 26f };
        var artifact = new Dictionary<string, object>
        {
            ["schema"] = "mayday-landing-residual-v1",
            ["mode"] = "residual",
            ["prior"] = "mayday-landing-observable-guidance-v1",
            ["residual_limits"] = new[] { 0.10f, 0.15f, 0.12f, 0.12f, 0f },
            ["stall_speeds"] = speeds,
            ["learner"] = Fixture(true),
        };
        string json = JsonSerializer.Serialize(artifact);
        LandingPolicy policy = LandingPolicy.Load(json);
        if (!policy.IsResidual || policy.IsPrior) throw new InvalidOperationException("Wrong residual mode.");

        // NumPy apply_residual_actions using a prior that exercises control clipping and ground brakes.
        var prior = new[] { 0.99f, -0.98f, 0.99f, 0.995f, 1f };
        for (int phase = 0; phase < ResidualGruExpected.Length; phase++)
        {
            var invalid = new[] { 0.99f, -0.98f, float.NaN, 0.995f, 1f };
            Rejects<ArgumentException>(() => policy.Predict(Observation(phase), phase, invalid));
            Near(policy.Predict(Observation(phase), phase, prior), ResidualGruExpected[phase], $"residual phase {phase}");
        }
        policy.Reset();
        Near(policy.Predict(Observation(4), 4, prior), new[] { 0.9760287f, -1f, 0.96789515f, 0.9988988f, 0f }, "residual reset");

        var automatic = LandingPolicy.Load(json);
        policy.Reset();
        for (int phase = 0; phase < ResidualGruExpected.Length; phase++)
        {
            float[] observation = Observation(phase);
            float[] explicitPrior = LandingPrior.Predict(observation, (LandingPhase)phase, speeds);
            Near(automatic.Predict(observation, phase), policy.Predict(observation, phase, explicitPrior), $"artifact-tuned prior, phase {phase}");
        }

        artifact["stall_speeds"] = new[] { 32f, 0f, 26f };
        Rejects<InvalidDataException>(() => LandingPolicy.Load(JsonSerializer.Serialize(artifact)));
        artifact["stall_speeds"] = speeds;
        artifact["residual_limits"] = new[] { 0.10f, 0.15f, 0.12f, 0.12f, 0.18f };
        Rejects<InvalidDataException>(() => LandingPolicy.Load(JsonSerializer.Serialize(artifact)));

        var priorArtifact = Fixture(false);
        priorArtifact["schema"] = "mayday-landing-observable-guidance-v1";
        priorArtifact["mode"] = "prior";
        priorArtifact["stall_speeds"] = speeds;
        priorArtifact.Remove("weights");
        LandingPolicy priorOnly = LandingPolicy.Load(JsonSerializer.Serialize(priorArtifact));
        if (!priorOnly.IsPrior || priorOnly.IsResidual) throw new InvalidOperationException("Wrong guidance mode.");
        Near(priorOnly.Predict(Observation(3), 3), LandingPrior.Predict(Observation(3), LandingPhase.Rollout, speeds), "prior-only rollout");
    }

    private static void Verify(bool recurrent, float[][] expected, float[] resetExpected)
    {
        LandingPolicy policy = LandingPolicy.Load(JsonSerializer.Serialize(Fixture(recurrent)));
        for (int phase = 0; phase < expected.Length; phase++)
        {
            var invalid = Observation(phase);
            invalid[^1] = float.NaN;
            Rejects<ArgumentException>(() => policy.Predict(invalid, phase));
            Near(policy.Predict(Observation(phase), phase), expected[phase], $"phase {phase}");
        }
        policy.Reset();
        Near(policy.Predict(Observation(4), 4), resetExpected, "step after reset");
        policy.Reset();
        Near(policy.Predict(Observation(0), 0), expected[0], "first step after full replay");
    }

    private static void Near(float[] actual, float[] expected, string step)
    {
        if (actual.Length != expected.Length)
            throw new InvalidOperationException($"Wrong action count at {step}.");
        for (int index = 0; index < actual.Length; index++)
        {
            if (!float.IsFinite(actual[index]) || Math.Abs(actual[index] - expected[index]) > 2e-6f)
                throw new InvalidOperationException($"Landing policy {step}, action {index}: {actual[index]:R}, expected {expected[index]:R}.");
        }
    }

    private static void Rejects<T>(Action action) where T : Exception
    {
        try
        {
            action();
        }
        catch (T)
        {
            return;
        }
        throw new InvalidOperationException($"Expected policy to throw {typeof(T).Name}.");
    }

    private static float[] Observation(int step) => Enumerable.Range(0, LandingFeatures.Names.Length)
        .Select(index => (float)((index * (step + 2) % 19 - 9) * 0.1)).ToArray();

    private static Dictionary<string, object> Fixture(bool recurrent)
    {
        static object Dense(int inputs, int outputs, double scale, int salt) => new
        {
            w = Enumerable.Range(0, inputs).Select(row => Enumerable.Range(0, outputs)
                .Select(col => scale * (((row + 3) * (col + 1) + salt) % 17 - 8)).ToArray()).ToArray(),
            b = Enumerable.Range(0, outputs).Select(col => scale * ((col % 5) - 2)).ToArray(),
        };

        int size = LandingFeatures.Names.Length;
        int head = recurrent ? 6 : 3;
        var weights = new Dictionary<string, object>
        {
            ["encoder"] = Dense(size + 5, 3, 0.0125, 3),
            ["air"] = Dense(head, 5, 0.09, 11),
            ["ground"] = Dense(head, 5, 0.065, 8),
        };
        if (recurrent)
        {
            weights["gru_input"] = Dense(3, 9, 0.08, 4);
            weights["gru_state"] = Dense(3, 9, 0.08, 1);
        }
        else
        {
            weights["hidden"] = Dense(3, 3, 0.07, 10);
        }
        return new Dictionary<string, object>
        {
            ["schema"] = "mayday-landing-policy-v1",
            ["architecture"] = recurrent ? "gru" : "mlp",
            ["feature_names"] = LandingFeatures.Names,
            ["actions"] = new[] { "pitch", "roll", "yaw", "throttle", "brake" },
            ["phases"] = new[] { "capture", "final", "flare", "rollout", "goaround" },
            ["hidden_size"] = 3,
            ["mean"] = Enumerable.Range(0, size).Select(index => (index % 7 - 3) * 0.05).ToArray(),
            ["std"] = Enumerable.Range(0, size).Select(index => 1.0 + (index % 5) * 0.125).ToArray(),
            ["weights"] = weights,
        };
    }
}
