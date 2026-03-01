# 将 Python App 打包为单个 EXE（PyInstaller / Windows / PowerShell）

> **目标**：生成 **单个可分发的 `StarFlowMaster.exe`**，并使用指定图标 `app_icon.ico`。

---

## 前置条件

1. **操作系统**：Windows
2. **项目目录要求**：
   - 包含 `main_app.py`
   - 包含 `app_icon.ico`（必须是 `.ico` 格式，非 jpg/png）
3. **建议工具**：PowerShell（确保所有命令在**项目目录**中执行）

---

## 步骤

### 1. 进入项目目录

```powershell
cd "你的项目目录"
dir main_app.py
dir app_icon.ico
```

---

### 2. 创建并激活虚拟环境

> **可选**：如果不想激活虚拟环境，后续命令可直接使用 `.\.venv\Scripts\python.exe ...` 的形式（推荐）。

1. 创建虚拟环境：

    ```powershell
    python -m venv .venv
    ```

2. 激活虚拟环境：

    ```powershell
    & .\.venv\Scripts\Activate.ps1
    ```

3. 验证是否成功进入虚拟环境（输出路径应包含 `.venv\Scripts\python.exe`）：

    ```powershell
    python -c "import sys; print(sys.executable)"
    ```

4. 若 PowerShell 报脚本执行策略限制，执行以下命令解除限制：

    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```

---

### 3. 安装依赖

1. 更新基础工具：

    ```powershell
    python -m pip install -U pip wheel setuptools
    ```

2. 安装打包所需依赖：

    ```powershell
    python -m pip install pyinstaller
    python -m pip install pyqt5 numpy opencv-python imageio imageio-ffmpeg sep
    ```

3. 验证依赖是否安装成功（应输出 `OK`）：

    ```powershell
    python -c "import PyQt5, numpy, cv2, imageio, imageio_ffmpeg, sep; print('OK')"
    ```

---

### 4. 打包为单个 EXE

> **注意**：务必使用虚拟环境中的 Python 调用 PyInstaller，避免依赖未正确打包。为了防止路径问题，建议直接使用 `.\.venv\Scripts\python.exe` 来调用 PyInstaller。

```powershell
.\.venv\Scripts\python.exe -m PyInstaller -y --noconfirm --clean `
  --name StarFlowMaster --windowed --onefile `
  --icon .\app_icon.ico `
  --hidden-import=imageio.v2 `
  --hidden-import=imageio_ffmpeg `
  --hidden-import=sep `
  --collect-all imageio `
  main_app.py
```

---

### 5. 验证打包结果

1. 生成的单文件 EXE 位于：

    ```
    dist\StarFlowMaster.exe
    ```

2. 测试运行：

    ```powershell
    .\dist\StarFlowMaster.exe
    ```

---

### 6. 分发应用

1. **分发文件**：仅需发送 `dist\StarFlowMaster.exe`。
2. **注意事项**：
   - `--onefile` 模式会在目标机器的临时目录解压依赖，用户无需额外文件。

---

### 7. 常见问题

1. **图标未刷新**：
   - 修改 EXE 文件名或重启资源管理器（清理 Windows 图标缓存）。

2. **提示 `imageio 未安装`**：
   - 确保使用虚拟环境中的 PyInstaller（参考第 4 步）。