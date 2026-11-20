import numpy as np
import matplotlib.pyplot as plt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder

# Grid
nx, ny = 128, 128
dx = dy = 0.1e-3

grid = kWaveGrid([nx, ny], [dx, dy])
grid.makeTime(1500)

# Medium: water
medium = kWaveMedium(sound_speed=1500, density=1000)

# Initial pressure source
source = kSource()
source.p0 = np.zeros((nx, ny))
source.p0[64, 64] = 1.0

# Record pressure at EVERY grid point
sensor_mask = np.ones((nx, ny), dtype=bool)
sensor = kSensor(mask=sensor_mask, record=["p"])

result = kspaceFirstOrder(
    grid, medium, source, sensor,
    backend="python",
    device="cpu",
    quiet=True,
)

print("Available keys:", list(result.keys()))

# Shape: (nx * ny, number of time steps)
p = result["p"]
print("Pressure data shape:", p.shape)

# Maximum absolute pressure at each grid point
p_peak = np.max(np.abs(p), axis=1).reshape(nx, ny)

# One instantaneous wavefield
step = p.shape[1] // 3
p_snapshot = p[:, step].reshape(nx, ny)

extent_mm = [0, nx * dx * 1000, 0, ny * dy * 1000]

# 1. Peak pressure map
plt.figure(figsize=(7, 6))
plt.imshow(p_peak.T, origin="lower",
           extent=extent_mm, cmap="inferno")
plt.colorbar(label="Peak absolute pressure (Pa)")
plt.title("Peak Pressure Distribution")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.tight_layout()
plt.savefig("pressure_map.png", dpi=150)
plt.close()

# 2. Wavefield snapshot
plt.figure(figsize=(7, 6))
plt.imshow(p_snapshot.T, origin="lower",
           extent=extent_mm, cmap="RdBu_r")
plt.colorbar(label="Pressure (Pa)")
plt.title(f"Wavefield at t = {step * grid.dt * 1e6:.2f} us")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.tight_layout()
plt.savefig("wave_snapshot.png", dpi=150)
plt.close()

# 3. Time waveform at grid point (90, 64)
sensor_index = 90 * ny + 64
pressure = p[sensor_index]
time_us = np.arange(p.shape[1]) * grid.dt * 1e6

plt.figure(figsize=(9, 4))
plt.plot(time_us, pressure)
plt.xlabel("Time (us)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure at (9.0 mm, 6.4 mm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("pressure_time.png", dpi=150)
plt.close()

print("Simulation completed.")
print("Saved pressure_map.png")
print("Saved wave_snapshot.png")
print("Saved pressure_time.png")
