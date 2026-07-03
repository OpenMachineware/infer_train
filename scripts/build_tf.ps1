# scripts/build_tf.ps1

Write-Host "🧠 Building InferTrain TensorFlow plugin..." -ForegroundColor Cyan

# 进入 TF 目录
Push-Location pybinds/infer_train_tf

Write-Host "1. Building wheel with maturin..." -ForegroundColor Yellow
maturin build --release

# 找 wheel
$WHEEL_PATH = Get-ChildItem "../../target/wheels/infer_train_tf-*.whl" | Select-Object -First 1
$WHEEL_NAME = $WHEEL_PATH.Name
$FIXED_WHEEL_NAME = $WHEEL_NAME -replace '\.whl$','_fixed.whl'
$PROJECT_ROOT = Resolve-Path "../.."
$FIXED_WHEEL_PATH = Join-Path $PROJECT_ROOT "target/wheels/$FIXED_WHEEL_NAME"

Write-Host "2. Extracting wheel..." -ForegroundColor Yellow
Remove-Item -Recurse -Force tmp -ErrorAction SilentlyContinue
Expand-Archive -Path $WHEEL_PATH -DestinationPath tmp

Write-Host "3. Replacing __init__.py..." -ForegroundColor Yellow
Copy-Item -Force __init__.py tmp/infer_train_tf/

Write-Host "4. Repacking wheel..." -ForegroundColor Yellow
Compress-Archive -Path tmp/* -DestinationPath $FIXED_WHEEL_PATH -Force

Write-Host "5. Cleaning up..." -ForegroundColor Yellow
Remove-Item -Recurse -Force tmp

Write-Host "✅ Fixed wheel: $FIXED_WHEEL_PATH" -ForegroundColor Green
Write-Host "   Install with: pip install $FIXED_WHEEL_PATH --force-reinstall"

Pop-Location
