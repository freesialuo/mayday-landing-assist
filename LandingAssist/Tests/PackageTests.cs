using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Text.Json;

namespace MaydayLanding.Tests;

internal static class PackageTests
{
    internal static void Check(string file, string? expectedPolicy)
    {
        using var package = ZipFile.OpenRead(file);
        using var manifestStream = (package.GetEntry("manifest.json") ?? throw new InvalidDataException("Missing manifest")).Open();
        using var manifest = JsonDocument.Parse(manifestStream);
        string name = manifest.RootElement.GetProperty("id").GetString() switch
        {
            "mayday-landing-auto" => "MaydayLandingAuto",
            "mayday-landing-advisor" => "MaydayLandingAdvisor",
            _ => throw new InvalidDataException("Unexpected install package ID")
        };
        string dllPath = $"Mods/{name}.dll";
        if (package.Entries.Count != 2 || package.GetEntry(dllPath) == null)
            throw new InvalidDataException("Packages must contain exactly manifest.json and their own mod DLL.");
        using var bytes = new MemoryStream();
        using (var entryStream = package.GetEntry(dllPath)!.Open()) entryStream.CopyTo(bytes);
        bytes.Position = 0;
        using var pe = new PEReader(bytes);
        var metadata = pe.GetMetadataReader();
        if (metadata.GetString(metadata.GetAssemblyDefinition().Name) != name) throw new InvalidDataException("Wrong DLL");
        var resources = metadata.ManifestResources.Select(id => metadata.GetManifestResource(id))
            .ToDictionary(r => metadata.GetString(r.Name), StringComparer.Ordinal);
        if (!resources.ContainsKey("MaydayLanding.Features")) throw new InvalidDataException("Missing feature schema");
        if (resources.TryGetValue("MaydayLanding.Policy.json", out var policy))
        {
            if (expectedPolicy == null) throw new InvalidDataException("Unexpected weights in package");
            var directory = pe.PEHeaders.CorHeader!.ResourcesDirectory;
            var blob = pe.GetSectionData(directory.RelativeVirtualAddress).GetContent().AsSpan().Slice((int)policy.Offset);
            int size = BinaryPrimitives.ReadInt32LittleEndian(blob);
            if (!blob.Slice(4, size).SequenceEqual(File.ReadAllBytes(expectedPolicy)))
                throw new InvalidDataException("Packaged policy differs from selected policy.");
        }
        else if (expectedPolicy != null) throw new InvalidDataException("Missing candidate weights in DLL");
        bool ownsInput = metadata.TypeDefinitions.Select(t => metadata.GetString(metadata.GetTypeDefinition(t).Name))
            .Contains("LandingInput");
        if (ownsInput != (name == "MaydayLandingAuto")) throw new InvalidDataException("Variant input ownership mismatch");
        if (name == "MaydayLandingAdvisor")
        {
            string[] forbidden = { "CmdSetThrottles", "CmdPilotAction", "CmdPilotInput" };
            if (metadata.MemberReferences.Select(id => metadata.GetString(metadata.GetMemberReference(id).Name))
                .Intersect(forbidden).Any() ||
                metadata.AssemblyReferences.Any(id => metadata.GetString(metadata.GetAssemblyReference(id).Name) == "0Harmony"))
                throw new InvalidDataException("Advisory assembly contains a write or patch dependency.");
        }
        Console.WriteLine($"PASS {name} package, resources and actuator boundary");
    }
}
