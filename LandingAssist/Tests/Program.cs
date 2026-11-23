// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using MaydayLanding;
using MaydayLanding.Tests;

if (args.Length is 2 or 3 && args[0] == "--check-package")
{
    PackageTests.Check(args[1], args.Length == 3 ? args[2] : null);
    return;
}

if (args.Length == 2 && args[0] == "--check-model")
{
    var model = LandingPolicy.Load(File.ReadAllText(args[1]));
    foreach (int phase in new[] { 0, 1, 2, 3, 4 })
    {
        var predicted = model.Predict(new float[LandingFeatures.Names.Length], phase);
        if (predicted.Length != 5 || predicted.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Invalid action.");
    }
    Console.WriteLine("PASS deployable policy");
    return;
}

DirectorTests.Run();
LandingPriorTests.Run();
ModelTests.Run();
ProbeTests.Run();
if (LandingFeatures.Names.Distinct().Count() != LandingFeatures.Names.Length) throw new Exception("Duplicate feature");
Console.WriteLine("PASS landing director, policy parity and feature schema");
