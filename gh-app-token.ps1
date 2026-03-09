# ~/.dotllm/gh-app-token.ps1
# Run with: . .\gh-app-token.ps1
# Sets $env:CLAUDE_GH_TOKEN to a fresh GitHub App installation token.
# Works on Windows PowerShell 5.1 (.NET Framework) and PowerShell 7+.
$ErrorActionPreference = 'Stop'

$AppId   = "2989778"
$PemPath = "$env:USERPROFILE\.dotllm\dotllm-claude-code-bot.2026-03-02.private-key.pem"

function ConvertTo-Base64Url([byte[]]$data) {
    [Convert]::ToBase64String($data).TrimEnd('=').Replace('+','-').Replace('/','_')
}

# Minimal DER parser — reads length field, advances index
function Read-DerLength([byte[]]$d, [ref]$i) {
    $b = $d[$i.Value++]
    if ($b -lt 0x80) { return [int]$b }
    $n = $b -band 0x7F; $len = 0
    for ($k = 0; $k -lt $n; $k++) { $len = ($len -shl 8) -bor $d[$i.Value++] }
    return $len
}

# Reads a DER INTEGER, strips leading sign byte, returns byte[]
function Read-DerInteger([byte[]]$d, [ref]$i) {
    if ($d[$i.Value++] -ne 0x02) { throw 'Expected DER INTEGER tag (0x02)' }
    $len   = Read-DerLength $d $i
    $bytes = New-Object byte[] $len
    [Array]::Copy($d, $i.Value, $bytes, 0, $len)
    $i.Value += $len
    if ($bytes.Length -gt 1 -and $bytes[0] -eq 0) {
        $trimmed = New-Object byte[] ($bytes.Length - 1)
        [Array]::Copy($bytes, 1, $trimmed, 0, $trimmed.Length)
        $bytes = $trimmed
    }
    return $bytes
}

# Pad byte[] to $len with leading zeros (RSACryptoServiceProvider requires exact prime lengths)
function PadLeft([byte[]]$b, [int]$len) {
    if ($b.Length -ge $len) { return $b }
    $r = New-Object byte[] $len
    [Array]::Copy($b, 0, $r, $len - $b.Length, $b.Length)
    return $r
}

# Parse PKCS#1 RSAPrivateKey DER into RSAParameters and import
function New-RsaFromPkcs1([byte[]]$der) {
    $i = 0
    if ($der[$i++] -ne 0x30) { throw 'Expected SEQUENCE at root' }
    Read-DerLength $der ([ref]$i) | Out-Null   # outer length
    Read-DerInteger $der ([ref]$i) | Out-Null  # version (INTEGER 0)

    $p          = New-Object System.Security.Cryptography.RSAParameters
    $p.Modulus  = Read-DerInteger $der ([ref]$i)
    $p.Exponent = Read-DerInteger $der ([ref]$i)
    $p.D        = Read-DerInteger $der ([ref]$i)
    $half       = [int][Math]::Ceiling($p.Modulus.Length / 2)
    $p.P        = PadLeft (Read-DerInteger $der ([ref]$i)) $half
    $p.Q        = PadLeft (Read-DerInteger $der ([ref]$i)) $half
    $p.DP       = PadLeft (Read-DerInteger $der ([ref]$i)) $half
    $p.DQ       = PadLeft (Read-DerInteger $der ([ref]$i)) $half
    $p.InverseQ = PadLeft (Read-DerInteger $der ([ref]$i)) $half

    $rsa = New-Object System.Security.Cryptography.RSACryptoServiceProvider
    $rsa.ImportParameters($p)
    return $rsa
}

# Load PKCS#1 PEM (GitHub always generates these — "BEGIN RSA PRIVATE KEY")
$pem = (Get-Content $PemPath -Raw) `
    -replace '-----BEGIN RSA PRIVATE KEY-----', '' `
    -replace '-----END RSA PRIVATE KEY-----',   '' `
    -replace '\s', ''
$rsa = New-RsaFromPkcs1 ([Convert]::FromBase64String($pem))

# Build JWT (exp = 9 min to avoid clock-skew rejection at the 10 min boundary)
$now      = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$header   = ConvertTo-Base64Url ([Text.Encoding]::UTF8.GetBytes('{"alg":"RS256","typ":"JWT"}'))
$payload  = ConvertTo-Base64Url ([Text.Encoding]::UTF8.GetBytes(
    "{`"iat`":$($now - 60),`"exp`":$($now + 540),`"iss`":`"$AppId`"}"))
$unsigned = "$header.$payload"

$sig = ConvertTo-Base64Url ($rsa.SignData(
    [Text.Encoding]::UTF8.GetBytes($unsigned),
    [Security.Cryptography.SHA256]::Create()))
$jwt = "$unsigned.$sig"

# Exchange JWT for installation token
$authHeader = @{ Authorization = "Bearer $jwt"; Accept = "application/vnd.github+json" }
$instId = (Invoke-RestMethod -Uri 'https://api.github.com/app/installations' -Headers $authHeader)[0].id
$token  = (Invoke-RestMethod -Method Post `
    -Uri "https://api.github.com/app/installations/$instId/access_tokens" `
    -Headers $authHeader).token

$env:CLAUDE_GH_TOKEN = $token
Write-Host "CLAUDE_GH_TOKEN set (expires 1h) - installation: $instId"
