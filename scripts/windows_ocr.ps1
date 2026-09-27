param(
    [string]$ImagePath,
    [string]$Language = 'zh-Hans-CN',
    [switch]$Server
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime]

function Wait-WinRt($Operation, [Type]$ResultType) {
    $Method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } |
        Select-Object -First 1
    $Task = $Method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $Task.GetAwaiter().GetResult()
}

$Engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Language))
if ($null -eq $Engine) { throw "Requested local OCR language unavailable: $Language" }
function Read-Image([string]$RequestedPath) {
$File = Wait-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync((Resolve-Path -LiteralPath $RequestedPath).Path)) ([Windows.Storage.StorageFile])
$Stream = Wait-WinRt ($File.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$Bitmap = $null
try {
    $Decoder = Wait-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($Stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $Bitmap = Wait-WinRt ($Decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    if ($Bitmap.PixelWidth -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or $Bitmap.PixelHeight -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension) {
        throw 'Image exceeds local OCR dimension limit'
    }
    $Result = Wait-WinRt ($Engine.RecognizeAsync($Bitmap)) ([Windows.Media.Ocr.OcrResult])
    $Lines = @($Result.Lines | ForEach-Object {
        @{
            text = $_.Text
            words = @($_.Words | ForEach-Object { @{
                text = $_.Text
                x = $_.BoundingRect.X; y = $_.BoundingRect.Y
                width = $_.BoundingRect.Width; height = $_.BoundingRect.Height
            } })
        }
    })
    @{ language = $Engine.RecognizerLanguage.LanguageTag; text = $Result.Text; lines = $Lines; confidence = $null; backend = 'windows_media_ocr_local' }
}
finally {
    if ($null -ne $Bitmap) { $Bitmap.Dispose() }
    $Stream.Dispose()
}
}

if ($Server) {
    while ($null -ne ($RequestLine = [Console]::ReadLine())) {
        $Request = $null
        try {
            $Request = $RequestLine | ConvertFrom-Json
            $Report = Read-Image ([string]$Request.path)
            @{ id = $Request.id; report = $Report } | ConvertTo-Json -Depth 10 -Compress
        }
        catch {
            @{ id = $Request.id; error = $_.Exception.Message } | ConvertTo-Json -Compress
        }
    }
}
else {
    if (-not $ImagePath) { throw 'ImagePath required outside server mode' }
    Read-Image $ImagePath | ConvertTo-Json -Depth 8 -Compress
}
