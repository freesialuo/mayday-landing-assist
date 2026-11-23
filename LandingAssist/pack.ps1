# Release: .\pack.ps1 -Candidate <policy.json>            -> <id>-<Version>.maydaymod, "(Experimental)", no probes
# Test:    .\pack.ps1 -Candidate <policy.json> -Probe     -> <id>-<Version>-probe.maydaymod, probes and auto-recording on
param([string] $Candidate, [switch] $Probe, [string] $Version = '0.2.0')
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$output = Join-Path $root '..\dist\landing'
[IO.Directory]::CreateDirectory($output) | Out-Null
$argsForPolicy = @()
$version = $Version
$label = ' (Experimental)'
$argsForPolicy += "-p:Version=$Version"
if ($Probe) { $argsForPolicy += '-p:LandingProbe=true'; $version = "$Version-probe"; $label = ' (Probe)' }
if (!$Candidate) {
    $defaultCandidate = Join-Path $root 'Models\candidate.json'
    if (Test-Path -LiteralPath $defaultCandidate -PathType Leaf) { $Candidate = $defaultCandidate }
}
dotnet run -c Release --project "$root\Tests\LandingTests.csproj"
if ($LASTEXITCODE -ne 0) { throw 'Landing verification failed' }
if ($Candidate) {
    $Candidate = (Get-Item -LiteralPath $Candidate).FullName
    dotnet run -c Release --no-build --project "$root\Tests\LandingTests.csproj" -- --check-model $Candidate
    if ($LASTEXITCODE -ne 0) { throw 'Invalid deployment model' }
    $argsForPolicy += "-p:LandingPolicyFile=$Candidate"
}
Add-Type -AssemblyName System.IO.Compression
foreach ($kind in @('Advisor','Auto')) {
    $assemblyName = "MaydayLanding$kind"
    dotnet build "$root\$kind\$assemblyName.csproj" -c Release @argsForPolicy
    if ($LASTEXITCODE -ne 0) { throw "$kind build failed" }
    $dll = Join-Path $root "$kind\bin\Release\net6.0\$assemblyName.dll"
    $id = 'mayday-landing-' + $kind.ToLowerInvariant()
    $manifest = [ordered]@{ schemaVersion=1; id=$id; name="Mayday Landing $kind$label"; version=$version }
    $buffer = New-Object IO.MemoryStream
    $archive = New-Object IO.Compression.ZipArchive($buffer, [IO.Compression.ZipArchiveMode]::Create, $true)
    try {
        $entry = $archive.CreateEntry('manifest.json').Open()
        try {
            $encoded = [Text.Encoding]::UTF8.GetBytes(($manifest | ConvertTo-Json -Compress))
            $entry.Write($encoded,0,$encoded.Length)
        } finally { $entry.Dispose() }
        $entry = $archive.CreateEntry("Mods/$assemblyName.dll").Open()
        $source = [IO.File]::OpenRead($dll)
        try { $source.CopyTo($entry) } finally { $source.Dispose(); $entry.Dispose() }
    } finally { $archive.Dispose() }
    $destination = Join-Path $output "$id-$version.maydaymod"
    try { [IO.File]::WriteAllBytes($destination, $buffer.ToArray()) } finally { $buffer.Dispose() }
    $verifyArgs = @('--check-package', $destination)
    if ($Candidate) { $verifyArgs += $Candidate }
    dotnet run -c Release --no-build --project "$root\Tests\LandingTests.csproj" -- @verifyArgs
    if ($LASTEXITCODE -ne 0) { throw "$kind package verification failed" }
    Write-Host "Packed $destination"
}
