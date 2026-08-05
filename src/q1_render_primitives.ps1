param(
    [Parameter(Mandatory=$true)][string]$Spec
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
$payload = Get-Content -LiteralPath $Spec -Raw -Encoding UTF8 | ConvertFrom-Json
$bitmap = [System.Drawing.Bitmap]::new([int]$payload.width, [int]$payload.height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([System.Drawing.Color]::White)

function Get-Color([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return [System.Drawing.Color]::Transparent }
    return [System.Drawing.ColorTranslator]::FromHtml($Value)
}

function Get-Font([int]$Size) {
    foreach ($name in @("Microsoft YaHei", "SimSun", "Arial")) {
        try { return [System.Drawing.Font]::new($name, [float]$Size) } catch { }
    }
    return [System.Drawing.Font]::new("Arial", [float]$Size)
}

try {
    foreach ($element in $payload.elements) {
        $kind = [string]$element.kind
        if ($kind -eq "text") {
            $font = Get-Font ([int]$element.size)
            $brush = [System.Drawing.SolidBrush]::new((Get-Color ([string]$element.color)))
            $format = [System.Drawing.StringFormat]::new()
            switch ([string]$element.align) {
                "center" { $format.Alignment = [System.Drawing.StringAlignment]::Center }
                "right" { $format.Alignment = [System.Drawing.StringAlignment]::Far }
                default { $format.Alignment = [System.Drawing.StringAlignment]::Near }
            }
            $angle = [float]$element.angle
            if ([Math]::Abs($angle) -gt 0.01) {
                $state = $graphics.Save()
                $graphics.TranslateTransform([float]$element.x, [float]$element.y)
                $graphics.RotateTransform($angle)
                $graphics.DrawString([string]$element.text, $font, $brush, 0.0, 0.0, $format)
                $graphics.Restore($state)
            } else {
                $graphics.DrawString([string]$element.text, $font, $brush, [float]$element.x, [float]$element.y, $format)
            }
            $format.Dispose(); $brush.Dispose(); $font.Dispose()
        } elseif ($kind -eq "line") {
            $pen = [System.Drawing.Pen]::new((Get-Color ([string]$element.color)), [float]$element.width)
            $graphics.DrawLine($pen, [float]$element.x1, [float]$element.y1, [float]$element.x2, [float]$element.y2)
            $pen.Dispose()
        } elseif ($kind -eq "rect" -or $kind -eq "ellipse") {
            $rect = [System.Drawing.RectangleF]::new([float]$element.x, [float]$element.y,
                [float]$element.width, [float]$element.height)
            $brush = [System.Drawing.SolidBrush]::new((Get-Color ([string]$element.fill)))
            if ($kind -eq "rect") { $graphics.FillRectangle($brush, $rect) }
            else { $graphics.FillEllipse($brush, $rect) }
            $brush.Dispose()
            if (-not [string]::IsNullOrWhiteSpace([string]$element.outline)) {
                $pen = [System.Drawing.Pen]::new((Get-Color ([string]$element.outline)), [float]$element.line_width)
                if ($kind -eq "rect") { $graphics.DrawRectangle($pen, $rect.X, $rect.Y, $rect.Width, $rect.Height) }
                else { $graphics.DrawEllipse($pen, $rect) }
                $pen.Dispose()
            }
        }
    }
    $output = [System.IO.Path]::GetFullPath([string]$payload.output)
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($output)) | Out-Null
    $bitmap.Save($output, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
