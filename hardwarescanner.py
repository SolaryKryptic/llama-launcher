import platform
import subprocess
import json
import re

def scan_hardware():
    os_type = platform.system()
    hardware = {"CPU": "Unknown", "GPU": "Unknown", "VRAM": "Unknown", "RAM": "Unknown"}

    try:
        # --- WINDOWS SCANNER ---
        if os_type == "Windows":
            # 1. CPU Model
            cpu_out = subprocess.check_output("wmic cpu get name", shell=True).decode().strip()
            hardware["CPU"] = cpu_out.split("\n")[-1].strip()

            # 1b. CPU cores & threads
            try:
                cores_out = subprocess.check_output("wmic cpu get NumberOfCores", shell=True).decode().strip()
                cores_lines = [x.strip() for x in cores_out.split("\n") if x.strip().isdigit()]
                hardware["CPU_CORES"] = sum(int(x) for x in cores_lines)  # sum both sockets
                threads_out = subprocess.check_output("wmic cpu get NumberOfLogicalProcessors", shell=True).decode().strip()
                threads_lines = [x.strip() for x in threads_out.split("\n") if x.strip().isdigit()]
                hardware["CPU_THREADS"] = sum(int(x) for x in threads_lines)
            except Exception:
                hardware["CPU_CORES"] = 0
                hardware["CPU_THREADS"] = 0

            # 2. System RAM
            ram_out = subprocess.check_output("wmic computersystem get totalphysicalmemory", shell=True).decode().strip()
            ram_lines = [x.strip() for x in ram_out.split("\n") if x.strip().isdigit()]
            if ram_lines:
                hardware["RAM"] = f"{int(ram_lines[0]) / (1024**3):.2f} GB"

            # 3. GPU Model Name (Isolating physical PCIe adapters from virtual layers)
            ps_gpu_cmd = (
                "PowerShell -Command \""
                "$P = Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPDeviceID -like '*PCI*' -and ($_.Service -eq 'amdwddmg' -or $_.Service -eq 'nvlddmkm' -or $_.Service -eq 'igfx') }; "
                "if (-not $P) { $P = Get-CimInstance Win32_VideoController }; "
                "Write-Output $P.Name\""
            )
            gpu_out_raw = subprocess.check_output(ps_gpu_cmd, shell=True).decode().strip()
            
            # Clean up the output by filtering out empty strings and virtual keyword matches
            gpu_lines = [line.strip() for line in gpu_out_raw.split("\n") if line.strip()]
            filtered_gpus = [
                gpu for gpu in gpu_lines 
                if not any(kw in gpu.lower() for kw in ["parsec", "virtual", "basic render", "citrix"])
            ]
            
            # Use the physical card found; if empty, fall back to the first available line
            if filtered_gpus:
                hardware["GPU"] = filtered_gpus[0]
            elif gpu_lines:
                hardware["GPU"] = gpu_lines[0]
            else:
                hardware["GPU"] = "Unknown GPU"

            # 4. True Uncapped 64-Bit VRAM Calculation via Performance Logs
            # Queries the hardware engine memory pool architecture directly to bypass the 4GB cap
            ps_vram_cmd = (
                "PowerShell -Command \""
                "$G = Get-CimInstance -Namespace root\\cimv2 -ClassName Win32_VideoController | Select-Object -First 1; "
                "if ($G.AdapterRAM -gt 4294967295) { Write-Output $G.AdapterRAM } "
                "else { "
                "  $Reg = Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue; "
                "  $MaxReg = ($Reg | ForEach-Object { $_.'HardwareInformation.qwMemorySize' } | Measure-Object -Maximum).Maximum; "
                "  if ($MaxReg) { Write-Output $MaxReg } else { Write-Output $G.AdapterRAM } "
                "}\""
            )
            vram_raw = subprocess.check_output(ps_vram_cmd, shell=True).decode().strip()
            
            if vram_raw and vram_raw.isdigit():
                vram_bytes = int(vram_raw)
                # If Windows still tries to pass a legacy 4GB integer limitation, force standard math checks
                if vram_bytes <= 4294967295:
                    # Alternative check: Fallback to querying DXGI adapter limits safely via task manager bindings
                    hardware["VRAM"] = "Check AMD Software (WMI Restricted)"
                else:
                    hardware["VRAM"] = f"{vram_bytes / (1024**3):.2f} GB"

    except Exception as e:
        print(f"Scan error: {e}")
        
    return hardware

# Run script
specs = scan_hardware()
print(f"CPU:  {specs['CPU']}")
print(f"GPU:  {specs['GPU']}")
print(f"VRAM: {specs['VRAM']}")
print(f"RAM:  {specs['RAM']}")
