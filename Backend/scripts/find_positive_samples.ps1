$base = "e:\Projects\Aerux_Web\AERUX-Frontend\Backend\outputs\test_set_inference_20260408_130441\samples"
$count = 0
Get-ChildItem -Path $base -Directory | ForEach-Object {
    $files = Get-ChildItem $_.FullName -File
    if ($files.Count -gt 2) {
        Write-Output "$($_.Name): $($files.Name -join ', ')"
        $count++
        if ($count -ge 5) { return }
    }
}
