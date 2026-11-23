// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace MaydayLanding;

/// <summary>Runs one player's exported landing policy and keeps its recurrent state.</summary>
public sealed class LandingPolicy
{
    private const string Schema = "mayday-landing-policy-v1";
    private const string ResidualSchema = "mayday-landing-residual-v1";
    private const string ResidualPrior = "mayday-landing-observable-guidance-v1";
    private const int Rollout = 3;
    private static readonly string[] Actions = { "pitch", "roll", "yaw", "throttle", "brake" };
    private static readonly string[] Phases = { "capture", "final", "flare", "rollout", "goaround" };
    private static readonly float[] ResidualLimits = { 0.10f, 0.15f, 0.12f, 0.12f, 0f };

    private readonly Dense? encoder;
    private readonly Dense? hiddenLayer;
    private readonly Dense? gruInput;
    private readonly Dense? gruState;
    private readonly Dense? air;
    private readonly Dense? ground;
    private readonly float[] mean = Array.Empty<float>();
    private readonly float[] std = Array.Empty<float>();
    private readonly float[] input = Array.Empty<float>();
    private readonly float[] encoded = Array.Empty<float>();
    private readonly float[] latent = Array.Empty<float>();
    private readonly float[] carry = Array.Empty<float>();
    private readonly float[] inputGates = Array.Empty<float>();
    private readonly float[] stateGates = Array.Empty<float>();
    private readonly float[]? priorStallSpeeds;

    public bool IsPrior => encoder == null;
    /// <summary>Flap-notch stall speeds of the embedded prior (residual policies only).</summary>
    public float[]? PriorStallSpeeds => (float[]?)priorStallSpeeds?.Clone();
    public bool IsResidual => encoder != null && priorStallSpeeds != null;

    private LandingPolicy(float[] priorStallSpeeds) => this.priorStallSpeeds = priorStallSpeeds;

    private LandingPolicy(JsonElement root, float[]? priorStallSpeeds)
    {
        if (root.GetProperty("schema").GetString() != Schema)
            throw new InvalidDataException("Unsupported landing policy schema.");

        ExpectNames(root, "feature_names", LandingFeatures.Names);
        ExpectNames(root, "actions", Actions);
        ExpectNames(root, "phases", Phases);
        this.priorStallSpeeds = priorStallSpeeds;

        bool recurrent = root.GetProperty("architecture").GetString() switch
        {
            "mlp" => false,
            "gru" => true,
            _ => throw new InvalidDataException("Unsupported landing policy architecture.")
        };

        int width = root.GetProperty("hidden_size").GetInt32();
        if (width < 1 || width > 256)
            throw new InvalidDataException("Invalid landing policy hidden size.");

        int features = LandingFeatures.Names.Length;
        mean = ReadVector(root.GetProperty("mean"), features, "mean");
        std = ReadVector(root.GetProperty("std"), features, "std");
        foreach (float deviation in std)
        {
            if (deviation <= 0f)
                throw new InvalidDataException("Landing policy standard deviations must be positive.");
        }

        input = new float[features + Phases.Length];
        encoded = new float[width];
        latent = new float[recurrent ? 2 * width : width];
        carry = recurrent ? new float[width] : Array.Empty<float>();
        inputGates = recurrent ? new float[3 * width] : Array.Empty<float>();
        stateGates = recurrent ? new float[3 * width] : Array.Empty<float>();

        JsonElement weights = root.GetProperty("weights");
        encoder = new Dense(weights, "encoder", input.Length, width);
        if (recurrent)
        {
            gruInput = new Dense(weights, "gru_input", width, inputGates.Length);
            gruState = new Dense(weights, "gru_state", width, stateGates.Length);
        }
        else
        {
            hiddenLayer = new Dense(weights, "hidden", width, width);
        }
        air = new Dense(weights, "air", latent.Length, Actions.Length);
        ground = new Dense(weights, "ground", latent.Length, Actions.Length);
    }

    public static LandingPolicy Load(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            string? schema = root.GetProperty("schema").GetString();
            if (schema == ResidualPrior)
            {
                if (root.GetProperty("mode").GetString() != "prior")
                    throw new InvalidDataException("Unsupported observable landing guidance mode.");
                ExpectNames(root, "feature_names", LandingFeatures.Names);
                ExpectNames(root, "actions", Actions);
                ExpectNames(root, "phases", Phases);
                return new LandingPolicy(ReadStallSpeeds(root));
            }
            float[]? stallSpeeds = null;
            if (schema == ResidualSchema)
            {
                if (root.GetProperty("mode").GetString() != "residual" ||
                    root.GetProperty("prior").GetString() != ResidualPrior)
                    throw new InvalidDataException("Unsupported landing residual mode or prior.");
                float[] limits = ReadVector(root.GetProperty("residual_limits"), Actions.Length, "residual limits");
                for (int i = 0; i < limits.Length; i++)
                {
                    if (limits[i] != ResidualLimits[i])
                        throw new InvalidDataException("Unsupported landing residual limits.");
                }
                stallSpeeds = ReadStallSpeeds(root);
                root = root.GetProperty("learner");
            }
            return new LandingPolicy(root, stallSpeeds);
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException
            or KeyNotFoundException or FormatException or OverflowException)
        {
            throw new InvalidDataException("Invalid landing policy JSON.", error);
        }
    }

    public void Reset() => Array.Clear(carry, 0, carry.Length);

    public float[] Predict(float[] observation, int phase)
    {
        PrepareInput(observation, phase);
        if (priorStallSpeeds == null) return PredictNetwork(phase);
        float[] prior = LandingPrior.Predict(observation, (LandingPhase)phase, priorStallSpeeds);
        ValidatePrior(prior);
        return IsPrior ? prior : PredictResidual(phase, prior);
    }

    public float[] Predict(float[] observation, int phase, float[] prior)
    {
        if (!IsResidual)
            throw new InvalidOperationException("Only a residual landing policy takes a separate guidance prior.");
        PrepareInput(observation, phase);
        ValidatePrior(prior);
        return PredictResidual(phase, prior);
    }

    private void PrepareInput(float[] observation, int phase)
    {
        if (observation == null || observation.Length != LandingFeatures.Names.Length)
            throw new ArgumentException("Observation length does not match the landing policy.", nameof(observation));
        if (phase < 0 || phase >= Phases.Length)
            throw new ArgumentOutOfRangeException(nameof(phase));

        for (int feature = 0; feature < observation.Length; feature++)
        {
            float value = observation[feature];
            if (!IsPrior)
            {
                value = (value - mean[feature]) / std[feature];
                input[feature] = value;
            }
            if (!float.IsFinite(value))
                throw new ArgumentException("Observation contains a non-finite feature.", nameof(observation));
        }
        if (IsPrior) return;
        Array.Clear(input, mean.Length, Phases.Length);
        input[mean.Length + phase] = 1f;
    }

    private float[] PredictNetwork(int phase)
    {
        encoder!.Apply(input, encoded);
        for (int index = 0; index < encoded.Length; index++)
            encoded[index] = MathF.Tanh(encoded[index]);

        if (hiddenLayer != null)
        {
            hiddenLayer.Apply(encoded, latent);
            for (int index = 0; index < latent.Length; index++)
                latent[index] = MathF.Tanh(latent[index]);
        }
        else
        {
            gruInput!.Apply(encoded, inputGates);
            gruState!.Apply(carry, stateGates);
            int width = carry.Length;
            Array.Copy(encoded, latent, width);
            for (int index = 0; index < width; index++)
            {
                float reset = Sigmoid(inputGates[index] + stateGates[index]);
                float update = Sigmoid(inputGates[index + width] + stateGates[index + width]);
                float candidate = MathF.Tanh(inputGates[index + 2 * width] + reset * stateGates[index + 2 * width]);
                latent[index + width] = (1f - update) * candidate + update * carry[index];
            }
        }

        var action = new float[Actions.Length];
        (phase == Rollout ? ground! : air!).Apply(latent, action);
        for (int index = 0; index < action.Length; index++)
        {
            action[index] = index < 3 ? MathF.Tanh(action[index]) : Sigmoid(action[index]);
            if (!float.IsFinite(action[index]))
                throw new InvalidDataException("Landing policy produced a non-finite action.");
        }
        if (hiddenLayer == null)
            Array.Copy(latent, carry.Length, carry, 0, carry.Length);
        return action;
    }

    private static void ValidatePrior(float[] prior)
    {
        if (prior == null || prior.Length != Actions.Length)
            throw new ArgumentException("Guidance prior must supply all landing controls.", nameof(prior));
        foreach (float value in prior)
        {
            if (!float.IsFinite(value))
                throw new ArgumentException("Guidance prior contains a non-finite control.", nameof(prior));
        }
    }

    private float[] PredictResidual(int phase, float[] prior)
    {
        float[] correction = PredictNetwork(phase);
        for (int i = 0; i < 3; i++)
            correction[i] = Math.Clamp(prior[i] + ResidualLimits[i] * correction[i], -1f, 1f);
        correction[3] = Math.Clamp(prior[3] + ResidualLimits[3] * (2f * correction[3] - 1f), 0f, 1f);
        correction[4] = phase == Rollout && prior[4] >= 0.5f ? 1f : 0f;
        return correction;
    }

    private static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-Math.Clamp(value, -80f, 80f)));

    private static float[] ReadStallSpeeds(JsonElement root)
    {
        float[] speeds = ReadVector(root.GetProperty("stall_speeds"), 3, "stall speeds");
        foreach (float speed in speeds)
        {
            if (speed <= 0f)
                throw new InvalidDataException("Landing guidance stall speeds must be positive.");
        }
        return speeds;
    }

    private static void ExpectNames(JsonElement root, string key, string[] expected)
    {
        JsonElement names = root.GetProperty(key);
        if (names.GetArrayLength() != expected.Length)
            throw new InvalidDataException($"Invalid landing policy {key} length.");

        for (int index = 0; index < expected.Length; index++)
        {
            if (names[index].GetString() != expected[index])
                throw new InvalidDataException($"Invalid landing policy {key} at position {index}.");
        }
    }

    private static float[] ReadVector(JsonElement values, int size, string key)
    {
        if (values.GetArrayLength() != size)
            throw new InvalidDataException($"Invalid landing policy {key} length.");
        var result = new float[size];
        for (int index = 0; index < size; index++)
        {
            result[index] = values[index].GetSingle();
            if (!float.IsFinite(result[index]))
                throw new InvalidDataException($"Invalid landing policy {key} value.");
        }
        return result;
    }

    private sealed class Dense
    {
        private readonly float[] matrix;
        private readonly float[] bias;

        public Dense(JsonElement weights, string name, int inputs, int outputs)
        {
            JsonElement layer = weights.GetProperty(name);
            JsonElement rows = layer.GetProperty("w");
            if (rows.GetArrayLength() != inputs)
                throw new InvalidDataException($"Invalid landing policy {name}.w row count.");

            matrix = new float[inputs * outputs];
            for (int row = 0; row < inputs; row++)
                Array.Copy(ReadVector(rows[row], outputs, name + ".w"), 0, matrix, row * outputs, outputs);
            bias = ReadVector(layer.GetProperty("b"), outputs, name + ".b");
        }

        public void Apply(float[] values, float[] result)
        {
            Array.Clear(result, 0, result.Length);
            int stride = result.Length;
            for (int row = 0; row < values.Length; row++)
            {
                float value = values[row];
                int start = row * stride;
                for (int column = 0; column < stride; column++)
                    result[column] += value * matrix[start + column];
            }
            for (int column = 0; column < stride; column++)
                result[column] += bias[column];
        }
    }
}
