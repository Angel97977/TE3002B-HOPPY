"""
plot_hoppy_log.py — Graficas de analisis para hop_cartesiano_log.csv (HOPPY)

Uso:
    python3 plot_hoppy_log.py
    python3 plot_hoppy_log.py --csv hop_cartesiano_log.csv --outdir plots
    python3 plot_hoppy_log.py --csv hop_cartesiano_log.csv --show   # ventanas interactivas

Genera, dentro de --outdir (default: plots/):
  01_posiciones.png         hip_pos / knee_pos vs t
  02_velocidades.png         vel_est vs vel_real (hip y knee) vs t
  03_torques.png             tau_hip / tau_knee vs t (con lineas TAU_MAX)
  04_contacto.png            Fn_sensor vs Fn_contact vs t, sombreado FLIGHT/STANCE
  05_trayectoria_pie.png      foot_x vs foot_z (espacio de trabajo, frame del hip)
  06_foot_z_world.png         foot_z_world vs t, linea de piso
  07_velocidad_pie.png        foot_vx / foot_vz vs t
  08_resumen.png              panel 2x2 con las señales mas relevantes
  09_tau_vs_omega_hip.png     torque vs velocidad articular (hip)
  10_tau_vs_omega_knee.png    torque vs velocidad articular (knee)
  11_grf.png                  GRF (Fx, Fz) en STANCE: medido (Real de MuJoCo) y perfil Bezier de referencia

Columnas esperadas (18+):
  t, phase, hip_pos, knee_pos, hip_vel_est, knee_vel_est, hip_vel_real,
  knee_vel_real, foot_x, foot_z, foot_vx, foot_vz, foot_z_world,
  tau_hip, tau_knee, Fn_sensor, Fn_contact
"""

import argparse
import csv
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TAU_MAX = 3.728

# Perfil Bezier de referencia GRF (de main.py, fase STANCE)
T_ST = 0.068
FX_BZ = np.array([0.0, 0.0, 24.0, 0.0, 0.0])
FZ_BZ = np.array([0.0, 20.0, 100.0, 0.0, 0.0])


def polyval_bz(alpha, s):
    n = len(alpha) - 1
    s = float(np.clip(s, 0.0, 1.0))
    return sum(a * math.comb(n, k) * s ** k * (1 - s) ** (n - k)
               for k, a in enumerate(alpha))


def load_csv(path):
    cols = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} esta vacio")

    # compatibilidad Fn_sensor -> Fn para logs viejos
    if "Fn" not in rows[0] and "Fn_sensor" in rows[0]:
        for r in rows:
            r["Fn"] = r["Fn_sensor"]

    for key in rows[0].keys():
        if key == "phase":
            cols[key] = [r[key] for r in rows]
        else:
            cols[key] = np.array([float(r[key]) for r in rows])
    return cols


def shade_phases(ax, t, phase):
    """Sombrea regiones donde phase == 'STANCE'."""
    in_stance = np.array([p == "STANCE" for p in phase])
    if not in_stance.any():
        return
    idx = list(np.where(np.diff(in_stance.astype(int)) != 0)[0] + 1)
    bounds = [0] + idx + [len(t)]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if in_stance[s]:
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color="orange", alpha=0.15, lw=0)


def plot_posiciones(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cols["t"], cols["hip_pos"], label="hip_pos", color="tab:blue")
    ax.plot(cols["t"], cols["knee_pos"], label="knee_pos", color="tab:red")
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("posicion [rad]")
    ax.set_title("Posiciones articulares (hip / knee)\nsombreado naranja = STANCE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "01_posiciones.png", show)


def plot_velocidades(cols, outdir, show):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(cols["t"], cols["hip_vel_real"], label="hip_vel_real", color="tab:blue", alpha=0.5)
    axes[0].plot(cols["t"], cols["hip_vel_est"], label="hip_vel_est (encoder)", color="tab:blue", lw=1.8)
    shade_phases(axes[0], cols["t"], cols["phase"])
    axes[0].set_ylabel("vel hip [rad/s]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Velocidad real (MuJoCo) vs estimada (encoder emulado, LP 40Hz)")

    axes[1].plot(cols["t"], cols["knee_vel_real"], label="knee_vel_real", color="tab:red", alpha=0.5)
    axes[1].plot(cols["t"], cols["knee_vel_est"], label="knee_vel_est (encoder)", color="tab:red", lw=1.8)
    shade_phases(axes[1], cols["t"], cols["phase"])
    axes[1].set_ylabel("vel knee [rad/s]")
    axes[1].set_xlabel("t [s]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, outdir, "02_velocidades.png", show)


def plot_torques(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cols["t"], cols["tau_hip"], label="tau_hip", color="tab:blue")
    ax.plot(cols["t"], cols["tau_knee"], label="tau_knee", color="tab:red")
    ax.axhline(TAU_MAX, color="k", ls="--", lw=1, label=f"+-TAU_MAX={TAU_MAX}")
    ax.axhline(-TAU_MAX, color="k", ls="--", lw=1)
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("torque [Nm]")
    ax.set_title("Torques de actuadores (saturacion en lineas punteadas)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "03_torques.png", show)


def plot_contacto(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    # Graficar la fuerza del sensor si existe
    if "Fn_sensor" in cols:
        ax.plot(cols["t"], cols["Fn_sensor"], color="tab:green", label="Fn_sensor (Resorte Emulado)")
    elif "Fn" in cols:
        ax.plot(cols["t"], cols["Fn"], color="tab:green", label="Fn (Sensor)")

    # Graficar la fuerza física real reportada por MuJoCo
    if "Fn_contact" in cols:
        ax.plot(cols["t"], cols["Fn_contact"], color="tab:red", alpha=0.6, lw=1.5, ls="--", label="Fn_contact (Física Real MuJoCo)")

    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Fuerza Normal [N]")
    ax.set_title("Comparativa de Contacto: Sensor Emulado vs Física Real")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "04_contacto.png", show)


def plot_trayectoria_pie(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(cols["foot_x"], cols["foot_z"], c=cols["t"], cmap="viridis", s=6)
    ax.plot(cols["foot_x"][0], cols["foot_z"][0], "go", ms=10, label="inicio")
    ax.plot(cols["foot_x"][-1], cols["foot_z"][-1], "rs", ms=10, label="fin")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("t [s]")
    ax.set_xlabel("foot_x [m]  (frame del hip)")
    ax.set_ylabel("foot_z [m]  (frame del hip)")
    ax.set_title("Trayectoria del pie en el plano sagital")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "05_trayectoria_pie.png", show)


def plot_foot_z_world(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cols["t"], cols["foot_z_world"], color="tab:purple")
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("foot_z_world [m]")
    ax.set_title("Altura del pie en el mundo (foot_z_world)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "06_foot_z_world.png", show)


def plot_velocidad_pie(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cols["t"], cols["foot_vx"], label="foot_vx", color="tab:blue")
    ax.plot(cols["t"], cols["foot_vz"], label="foot_vz", color="tab:red")
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("velocidad [m/s]")
    ax.set_title("Velocidad cartesiana del pie (J * dq_est)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, outdir, "07_velocidad_pie.png", show)


def _tau_vs_omega(ax, omega, tau, label, color, t):
    sc = ax.scatter(omega, tau, c=t, cmap="viridis", s=8)
    ax.axhline(TAU_MAX, color="k", ls="--", lw=1)
    ax.axhline(-TAU_MAX, color="k", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"omega_{label} [rad/s]  (vel_real)")
    ax.set_ylabel(f"tau_{label} [Nm]")
    ax.set_title(f"Torque vs velocidad — {label}\n(lineas punteadas = +-TAU_MAX={TAU_MAX})")
    ax.grid(alpha=0.3)
    return sc


def plot_tau_vs_omega_hip(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = _tau_vs_omega(ax, cols["hip_vel_real"], cols["tau_hip"], "hip", "tab:blue", cols["t"])
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("t [s]")
    fig.tight_layout()
    _save(fig, outdir, "09_tau_vs_omega_hip.png", show)


def plot_tau_vs_omega_knee(cols, outdir, show):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = _tau_vs_omega(ax, cols["knee_vel_real"], cols["tau_knee"], "knee", "tab:red", cols["t"])
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("t [s]")
    fig.tight_layout()
    _save(fig, outdir, "10_tau_vs_omega_knee.png", show)


def plot_grf(cols, outdir, show):
    """GRF (Fx, Fz): componentes medidas (fuerza real si existe) vs perfil Bezier de referencia."""
    t = cols["t"]
    phase = cols["phase"]
    
    # Fase 3 de rúbrica: Priorizar Fn_contact (Fuerza real física)
    if "Fn_contact" in cols:
        fn = cols["Fn_contact"]
        label_fz = "Fz medido (Física Real MuJoCo)"
    else:
        fn = cols.get("Fn_sensor", cols.get("Fn", np.zeros_like(t)))
        label_fz = "Fz medido (Sensor Emulado)"

    fz_meas = fn.copy()
    fx_meas = np.zeros_like(fn)

    fx_ref = np.full_like(fn, np.nan)
    fz_ref = np.full_like(fn, np.nan)
    t_stance_start = None
    
    for i in range(len(t)):
        if phase[i] == "STANCE":
            if t_stance_start is None or phase[i - 1] != "STANCE":
                t_stance_start = t[i]
            s = (t[i] - t_stance_start) / T_ST
            fx_ref[i] = polyval_bz(FX_BZ, s)
            fz_ref[i] = polyval_bz(FZ_BZ, s)
        else:
            t_stance_start = None

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(t, fx_meas, label="Fx medido (=0, no disponible en log directo)", color="tab:blue", alpha=0.5)
    ax.plot(t, fx_ref, label="Fx referencia (Bezier FX_BZ)", color="tab:blue", lw=1.8, ls="--")
    shade_phases(ax, t, phase)
    ax.set_ylabel("Fx [N]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("GRF — componente X (sombreado naranja = STANCE)")

    ax = axes[1]
    ax.plot(t, fz_meas, label=label_fz, color="tab:green")
    ax.plot(t, fz_ref, label="Fz referencia (Bezier FZ_BZ)", color="tab:green", lw=1.8, ls="--")
    shade_phases(ax, t, phase)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Fz [N]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("GRF — componente Z (Curva Teórica vs. Medición Real)")

    fig.tight_layout()
    _save(fig, outdir, "11_grf.png", show)


def plot_resumen(cols, outdir, show):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(cols["t"], cols["hip_pos"], label="hip_pos", color="tab:blue")
    ax.plot(cols["t"], cols["knee_pos"], label="knee_pos", color="tab:red")
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_title("Posiciones articulares")
    ax.set_xlabel("t [s]"); ax.set_ylabel("rad")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(cols["t"], cols["tau_hip"], label="tau_hip", color="tab:blue")
    ax.plot(cols["t"], cols["tau_knee"], label="tau_knee", color="tab:red")
    ax.axhline(TAU_MAX, color="k", ls="--", lw=1)
    ax.axhline(-TAU_MAX, color="k", ls="--", lw=1)
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_title("Torques")
    ax.set_xlabel("t [s]"); ax.set_ylabel("Nm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    fn_key = "Fn_contact" if "Fn_contact" in cols else "Fn_sensor"
    ax.plot(cols["t"], cols[fn_key], color="tab:green")
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_title(f"Fuerza de contacto ({fn_key})")
    ax.set_xlabel("t [s]"); ax.set_ylabel("N")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(cols["t"], cols["foot_z_world"], color="tab:purple")
    shade_phases(ax, cols["t"], cols["phase"])
    ax.set_title("Altura del pie en el mundo")
    ax.set_xlabel("t [s]"); ax.set_ylabel("foot_z_world [m]")
    ax.grid(alpha=0.3)

    fig.suptitle("HOPPY — resumen de simulacion (naranja = STANCE)", y=1.02, fontsize=13)
    fig.tight_layout()
    _save(fig, outdir, "08_resumen.png", show)


def _save(fig, outdir, name, show):
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  -> {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def print_stats(cols):
    print("\n=== Estadisticas rapidas ===")
    print(f"Duracion total      : {cols['t'][-1]:.3f} s   ({len(cols['t'])} muestras)")

    phases = cols["phase"]
    n_stance = sum(1 for p in phases if p == "STANCE")
    n_flight = sum(1 for p in phases if p == "FLIGHT")
    print(f"Muestras FLIGHT/STANCE : {n_flight} / {n_stance}")

    td = lo = 0
    for i in range(1, len(phases)):
        if phases[i - 1] == "FLIGHT" and phases[i] == "STANCE":
            td += 1
        elif phases[i - 1] == "STANCE" and phases[i] == "FLIGHT":
            lo += 1
    print(f"Touchdowns (TD)      : {td}")
    print(f"Lift-offs  (LO)      : {lo}")
    
    # Stats prioritizando Fn_contact
    if "Fn_contact" in cols:
        print(f"\nFn_contact       : min={cols['Fn_contact'].min():.4f}  max={cols['Fn_contact'].max():.4f}  "
              f"mean={cols['Fn_contact'].mean():.4f} N")
    elif "Fn_sensor" in cols:
         print(f"\nFn_sensor        : min={cols['Fn_sensor'].min():.4f}  max={cols['Fn_sensor'].max():.4f}  "
              f"mean={cols['Fn_sensor'].mean():.4f} N")

    print(f"foot_z_world     : min={cols['foot_z_world'].min():.4f}  "
          f"max={cols['foot_z_world'].max():.4f}  final={cols['foot_z_world'][-1]:.4f} m")

    for name in ("tau_hip", "tau_knee"):
        v = cols[name]
        sat = np.sum(np.abs(v) >= TAU_MAX * 0.999)
        print(f"{name:<16} : min={v.min():.4f}  max={v.max():.4f}  "
              f"saturado={sat}/{len(v)} muestras")

    for name in ("hip_vel_real", "knee_vel_real"):
        v = cols[name]
        print(f"{name:<16} : min={v.min():.4f}  max={v.max():.4f} rad/s")

    for name in ("hip_pos", "knee_pos"):
        v = cols[name]
        print(f"{name:<16} : min={v.min():.4f}  max={v.max():.4f}  final={v[-1]:.4f} rad")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graficas de hop_cartesiano_log.csv")
    ap.add_argument("--csv", default="hop_cartesiano_log.csv", help="archivo CSV de log")
    ap.add_argument("--outdir", default="plots", help="carpeta de salida para PNGs")
    ap.add_argument("--show", action="store_true", help="mostrar ventanas interactivas")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Cargando {args.csv} ...")
    cols = load_csv(args.csv)

    print_stats(cols)

    print(f"\nGenerando graficas en '{args.outdir}/' ...")
    plot_posiciones(cols, args.outdir, args.show)
    plot_velocidades(cols, args.outdir, args.show)
    plot_torques(cols, args.outdir, args.show)
    plot_contacto(cols, args.outdir, args.show)
    plot_trayectoria_pie(cols, args.outdir, args.show)
    plot_foot_z_world(cols, args.outdir, args.show)
    plot_velocidad_pie(cols, args.outdir, args.show)
    plot_tau_vs_omega_hip(cols, args.outdir, args.show)
    plot_tau_vs_omega_knee(cols, args.outdir, args.show)
    plot_grf(cols, args.outdir, args.show)
    plot_resumen(cols, args.outdir, args.show)

    print("\nListo.")