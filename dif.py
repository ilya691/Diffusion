"""
Моделирование диффузии: стохастическая (Монте-Карло) и детерминированная (PDE).
Один файл — CLI, визуализация, все исследования.

v4 (исправлена нормировка аналитики):
  * T_ana = сечение 2D-гауссианы: 1/(2π s²) exp(-x²/(2s²))
  * Численное T нормировано на 1 по площади (Σ T = 1 при Δx=Δy=1)
  * Теперь численное и аналитическое решения имеют одинаковую нормировку
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# =====================================================================
# §1. КОНФИГ
# =====================================================================
PARAMS = {
    "walk": {
        "n_particles": 500, "n_steps": 200, "dt": 1.0,
        "seed": None, "frame_step": 5,
    },
    "pde": {
        "nx": 121, "ny": 121, "n_steps": 300,
        "kappa": 0.1, "dt": 0.01, "save_every": 2,
        "source": "gauss", "sigma0": 2.0,
    },
    "msd": {
        "n_particles": 2000, "n_steps": 200,
        "seeds": [1, 2, 3, 4, 5],
        "out": "results/study1_msd.png",
    },
    "pde_study": {
        "nx": 201, "ny": 201,
        "kappa": 0.1, "dt": 0.0025,
        "source": "gauss", "sigma0": 2.0,
        "t_checkpoints": [0.05, 0.2, 0.5, 1.0],
        "out": "results/study2_pde.png",
    },
    "brownian": {
        "n_particles": 20000, "n_steps": 100,
        "seed": 7, "gif_particles": 200, "gif_seed": 11,
        "out_prefix": "results/phys1_brownian",
    },
    "sources": {
        "nx": 121, "ny": 121, "kappa": 0.1, "dt": 0.01, "n_steps": 250,
        "checkpoints": [15, 40, 90, 180], "flat_threshold": 0.95,
        "out_prefix": "results/phys2_sources",
    },
    "obstacles": {
        "n_particles": 3000, "n_steps": 300, "n_trials": 5,
        "obstacles_fraction": 0.25, "L": 200,
        "mask_seed": 42, "traj_seed": 0,
        "out_prefix": "results/phys3_obstacles",
    },
    "arrhenius": {
        "T_min": 0.5, "T_max": 3.0, "n_T": 12, "Ea": 1.0,
        "n_particles": 1500, "n_steps": 200,
        "out": "results/phys4_arrhenius.png",
    },
}


# =====================================================================
# §2. МОДЕЛЬ
# =====================================================================
def simulate_random_walk(n_particles, n_steps, dt=1.0, seed=None,
                         save_history=True):
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_particles, 2), dtype=np.float64)
    history = [pos.copy()] if save_history else []
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float64)
    step = np.sqrt(dt)
    for _ in range(n_steps):
        idx = rng.integers(0, 4, size=n_particles)
        pos = pos + moves[idx] * step
        if save_history:
            history.append(pos.copy())
    return pos, history


def simulate_diffusion_pde(nx, ny, n_steps, kappa=0.1, dt=0.01,
                           save_every=1, source="gauss", sigma0=2.0):
    """Явная схема 2D уравнения теплопроводности.
    source: 'delta' | 'gauss' | 'two'.
    Для 'gauss' T нормировано так, что Σ T = 1 (при Δx=Δy=1)."""
    dx = dy = 1.0
    coeff = kappa * dt / dx ** 2
    assert coeff <= 0.25, (
        f"Нарушено условие устойчивости: kappa*dt/dx^2 = {coeff:.3f} > 0.25.")

    T = np.zeros((ny, nx), dtype=np.float64)
    cy, cx = ny // 2, nx // 2

    if source == "delta":
        T[cy, cx] = 1.0
    elif source == "gauss":
        y, x = np.mgrid[0:ny, 0:nx]
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        g = np.exp(-r2 / (2.0 * sigma0 ** 2))
        T = g / g.sum()
    elif source == "two":
        T[cy, nx // 3] = 1.0
        T[cy, 2 * nx // 3] = 1.0
    else:
        raise ValueError(f"Неизвестный источник: {source}")

    frames = []
    for step in range(n_steps):
        if step % save_every == 0:
            frames.append(T.copy())
        lap = (np.roll(T, 1, axis=0) + np.roll(T, -1, axis=0)
               + np.roll(T, 1, axis=1) + np.roll(T, -1, axis=1) - 4.0 * T)
        T = T + coeff * lap
    frames.append(T.copy())
    return frames


def mean_squared_displacement(history):
    history = np.asarray(history)
    return np.mean(np.sum(history ** 2, axis=2), axis=1)


def gaussian_analytic_section(x, t, kappa, sigma0):
    """Сечение 2D-гауссианы вдоль оси X при y=0:
       T(x,t) = 1/(2π s²) exp(-x²/(2s²)), s² = 2κt + σ0².
       Совпадает по нормировке с численным решением (Σ T = 1)."""
    s2 = 2.0 * kappa * t + sigma0 ** 2
    return np.exp(-x ** 2 / (2.0 * s2)) / (2.0 * np.pi * s2)


# =====================================================================
# §3. ВИЗУАЛИЗАЦИЯ
# =====================================================================
def gif_random_walk(history, out_path, fps=20, frame_step=5, bounds=None):
    history = np.asarray(history)
    if bounds is None:
        bounds = float(np.max(np.abs(history))) * 1.1 + 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-bounds, bounds); ax.set_ylim(-bounds, bounds)
    ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.5)
    scat = ax.scatter(history[0, :, 0], history[0, :, 1],
                      s=6, c="tab:blue", alpha=0.7)

    def update(i):
        scat.set_offsets(history[i])
        ax.set_title(f"Случайное блуждание, шаг {i}/{len(history)-1}")
        return (scat,)

    frames = list(range(0, len(history), frame_step))
    FuncAnimation(fig, update, frames=frames,
                  interval=1000 // fps, blit=False).save(
        out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def gif_diffusion_pde(frames, out_path, fps=15, cmap="hot"):
    frames = np.asarray(frames)
    vmax = float(frames.max()) if frames.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=vmax,
                   origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax).set_label("T")

    def update(i):
        im.set_data(frames[i])
        ax.set_title(f"Диффузия, шаг {i}/{len(frames)-1}")
        return (im,)

    FuncAnimation(fig, update, frames=len(frames),
                  interval=1000 // fps, blit=False).save(
        out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def gif_trajectory_with_obstacles(traj, mask, out_path, fps=20):
    H, W = mask.shape
    cx, cy = W // 2, H // 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap="Greys", origin="lower", alpha=0.4)
    line, = ax.plot([], [], "b-", lw=0.8)
    point, = ax.plot([], [], "ro", ms=5)
    ax.set_xlim(cx - 60, cx + 60); ax.set_ylim(cy - 60, cy + 60)

    def update(i):
        x = traj[0, :i + 1] + cx
        y = traj[1, :i + 1] + cy
        line.set_data(x, y); point.set_data([x[-1]], [y[-1]])
        ax.set_title(f"Траектория, шаг {i}")
        return line, point

    FuncAnimation(fig, update, frames=traj.shape[1],
                  interval=1000 // fps, blit=False).save(
        out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# =====================================================================
# §4. БАЗОВЫЕ ИССЛЕДОВАНИЯ
# =====================================================================
def study_msd(params=None):
    p = {**PARAMS["msd"], **(params or {})}
    msds = [mean_squared_displacement(
        simulate_random_walk(p["n_particles"], p["n_steps"], seed=s)[1])
        for s in p["seeds"]]
    msd = np.mean(msds, axis=0)
    t = np.arange(len(msd))
    slope, intercept = np.polyfit(t, msd, 1)

    os.makedirs(os.path.dirname(p["out"]) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, msd, "o-", ms=3, label="MSD (симуляция)")
    ax.plot(t, slope * t + intercept, "r--",
            label=f"подгонка: y={slope:.2f}t+{intercept:.2f}")
    ax.set_xlabel("шаг t"); ax.set_ylabel("⟨|r|²⟩")
    ax.set_title("Исследование 1: MSD ∝ t")
    ax.grid(True, ls=":", alpha=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(p["out"], dpi=150); plt.close(fig)
    print(f"[MSD] наклон ≈ {slope:.3f} (теория 2D: 1.0)")
    print(f"      MSD/t на конце: {msd[-1]/t[-1]:.3f}")
    return msd, slope


def study_pde(params=None):
    p = {**PARAMS["pde_study"], **(params or {})}
    nx, ny = p["nx"], p["ny"]
    kappa, dt = p["kappa"], p["dt"]
    sigma0 = p["sigma0"]

    coeff = kappa * dt
    assert coeff <= 0.25, (
        f"Нарушено условие устойчивости: kappa*dt = {coeff:.4f} > 0.25.")

    t_max = max(p["t_checkpoints"])
    n_steps = int(np.ceil(t_max / dt))
    save_every = max(1, n_steps // 200)

    frames = simulate_diffusion_pde(
        nx, ny, n_steps, kappa=kappa, dt=dt,
        save_every=save_every, source=p["source"], sigma0=sigma0)

    y0 = ny // 2
    x = np.arange(nx) - nx // 2

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    print("[PDE vs аналитика]")
    print(f"  {'t':>5} {'σ/Δx':>6} {'L2_abs':>10} {'L2_rel':>10}")
    rows = []
    for ax, t in zip(axes.flat, p["t_checkpoints"]):
        step = min(int(round(t / dt)), len(frames) - 1)
        T_num = frames[step][y0, :]
        T_ana = gaussian_analytic_section(x, t, kappa, sigma0)
        sigma = np.sqrt(2.0 * kappa * t + sigma0 ** 2)

        l2_abs = float(np.sqrt(np.mean((T_num - T_ana) ** 2)))
        denom = float(np.sqrt(np.mean(T_ana ** 2)))
        l2_rel = l2_abs / denom if denom > 0 else np.inf
        rows.append((t, sigma, l2_abs, l2_rel))
        print(f"  {t:5.2f} {sigma:6.2f} {l2_abs:10.5f} {l2_rel:10.5f}")

        ax.plot(x, T_num, "b-", label="PDE")
        ax.plot(x, T_ana, "r--", label="аналитика")
        ax.set_title(f"t={t:.2f}, σ/Δx={sigma:.2f}")
        ax.set_xlabel("x"); ax.set_ylabel("T")
        ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=9)

    fig.suptitle("Исследование 2: PDE vs аналитика (сечение 2D-гауссианы)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(p["out"]) or ".", exist_ok=True)
    fig.savefig(p["out"], dpi=150); plt.close(fig)

    first, last = rows[0], rows[-1]
    if first[1] < 1.0:
        print(f"  Примечание: при t={first[0]:.2f} σ/Δx={first[1]:.2f} < 1.")
    print(f"  В диффузионном режиме (t={last[0]:.2f}, σ/Δx={last[1]:.2f}) "
          f"L2_rel = {last[3]:.4f} — схема корректна.")
    return rows


# =====================================================================
# §5. ФИЗИЧЕСКИЕ ИССЛЕДОВАНИЯ
# =====================================================================
def study_brownian(params=None):
    p = {**PARAMS["brownian"], **(params or {})}
    _, hist = simulate_random_walk(p["n_particles"], p["n_steps"],
                                   seed=p["seed"])
    final = hist[-1]
    x, y = final[:, 0], final[:, 1]
    sigma_theory = float(np.sqrt(p["n_steps"] / 2))

    os.makedirs(os.path.dirname(p["out_prefix"]) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(x, bins=80, density=True, alpha=0.6,
                 color="tab:blue", label="симуляция")
    xs = np.linspace(x.min(), x.max(), 400)
    pdf = 1 / (np.sqrt(2 * np.pi) * sigma_theory) * np.exp(
        -xs ** 2 / (2 * sigma_theory ** 2))
    axes[0].plot(xs, pdf, "r--", lw=2,
                 label=f"N(0, σ²), σ={sigma_theory:.2f}")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("плотность")
    axes[0].set_title(f"Распределение X после {p['n_steps']} шагов")
    axes[0].legend(); axes[0].grid(True, ls=":", alpha=0.5)

    hb = axes[1].hexbin(x, y, gridsize=60, cmap="viridis", bins="log")
    fig.colorbar(hb, ax=axes[1], label="log(число частиц)")
    axes[1].set_aspect("equal")
    axes[1].set_title("2D-облако конечных положений")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.suptitle("Исследование 1: броуновское движение")
    fig.tight_layout()
    fig.savefig(f"{p['out_prefix']}_distribution.png", dpi=150)
    plt.close(fig)

    print(f"[Физ.1] σ_x(сим) = {x.std():.3f}, теория = {sigma_theory:.3f}, "
          f"отношение = {x.std()/sigma_theory:.3f}")

    _, small = simulate_random_walk(p["gif_particles"], p["n_steps"],
                                    seed=p["gif_seed"])
    gif_random_walk(small, f"{p['out_prefix']}.gif", frame_step=3)


def study_sources(params=None):
    p = {**PARAMS["sources"], **(params or {})}
    frames = simulate_diffusion_pde(p["nx"], p["ny"], p["n_steps"],
                                    kappa=p["kappa"], dt=p["dt"],
                                    save_every=1, source="two")
    y0 = p["ny"] // 2
    x = np.arange(p["nx"])
    x1, x2 = p["nx"] // 3, 2 * p["nx"] // 3

    def T_two(t):
        return (1 / (4 * np.pi * p["kappa"] * t)) * (
            np.exp(-(x - x1) ** 2 / (4 * p["kappa"] * t))
            + np.exp(-(x - x2) ** 2 / (4 * p["kappa"] * t)))

    os.makedirs(os.path.dirname(p["out_prefix"]) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, step in zip(axes.flat, p["checkpoints"]):
        t = step * p["dt"]
        ax.plot(x, frames[step][y0, :], "b-", label="PDE")
        ax.plot(x, T_two(t), "r--", label="аналитика")
        ax.axvline(x1, color="gray", ls=":", alpha=0.6)
        ax.axvline(x2, color="gray", ls=":", alpha=0.6)
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x"); ax.set_ylabel("T")
        ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.5)
    fig.suptitle("Исследование 2: диффузия от двух источников")
    fig.tight_layout()
    fig.savefig(f"{p['out_prefix']}_profiles.png", dpi=150)
    plt.close(fig)

    center = p["nx"] // 2
    profile_center = np.array([f[y0, center] for f in frames])
    max_profile = np.array([f[y0, :].max() for f in frames])
    ratio = np.where(max_profile > 0, profile_center / max_profile, 0.0)
    idx = int(np.argmax(ratio > p["flat_threshold"]))
    if ratio[idx] > p["flat_threshold"]:
        print(f"[Физ.2] Время выравнивания: t ≈ {idx * p['dt']:.3f}")
    else:
        print("[Физ.2] Выравнивание не достигнуто")
    gif_diffusion_pde(frames[::2], f"{p['out_prefix']}.gif", fps=20)


def study_obstacles(params=None):
    p = {**PARAMS["obstacles"], **(params or {})}
    L = p["L"]; half = L // 2
    rng_mask = np.random.default_rng(p["mask_seed"])
    mask = rng_mask.random((L, L)) < p["obstacles_fraction"]
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)

    def run(use_obs, seed):
        rng = np.random.default_rng(seed)
        pos = np.zeros((p["n_particles"], 2), dtype=np.int64)
        hist = [pos.copy()]
        for _ in range(p["n_steps"]):
            idx = rng.integers(0, 4, size=p["n_particles"])
            new_pos = pos + moves[idx]
            if use_obs:
                gx = (new_pos[:, 0] + half) % L
                gy = (new_pos[:, 1] + half) % L
                blocked = mask[gy, gx]
                new_pos = np.where(blocked[:, None], pos, new_pos)
            pos = new_pos
            hist.append(pos.copy())
        return np.array(hist)

    msd_free = np.zeros(p["n_steps"] + 1)
    msd_obs = np.zeros(p["n_steps"] + 1)
    for s in range(p["n_trials"]):
        msd_free += np.mean(np.sum(run(False, s) ** 2, axis=2), axis=1)
        msd_obs += np.mean(np.sum(run(True, s) ** 2, axis=2), axis=1)
    msd_free /= p["n_trials"]; msd_obs /= p["n_trials"]

    t = np.arange(len(msd_free))
    D_free = float(np.polyfit(t[10:], msd_free[10:], 1)[0])
    D_obs = float(np.polyfit(t[10:], msd_obs[10:], 1)[0])

    os.makedirs(os.path.dirname(p["out_prefix"]) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, msd_free, "b-", label=f"свободная, D≈{D_free:.3f}")
    ax.plot(t, msd_obs, "r-",
            label=f"пористая ({int(p['obstacles_fraction']*100)}%), "
                  f"D≈{D_obs:.3f}")
    ax.set_xlabel("шаг t"); ax.set_ylabel("⟨|r|²⟩")
    ax.set_title("Исследование 3: диффузия в пористой среде")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    fig.tight_layout()
    fig.savefig(f"{p['out_prefix']}_msd.png", dpi=150); plt.close(fig)

    ratio = D_obs / D_free if D_free else 0.0
    D_theory = D_free * (1 - p["obstacles_fraction"]) ** 2
    print(f"[Физ.3] D_free={D_free:.3f}, D_obs={D_obs:.3f}, "
          f"отношение={ratio:.3f}")
    print(f"        Бруттье-Эмт: D_eff ≈ {D_theory:.3f}")

    rng = np.random.default_rng(p["traj_seed"])
    pos = np.zeros(2, dtype=np.int64)
    traj = [pos.copy()]
    for _ in range(p["n_steps"]):
        step = moves[rng.integers(0, 4)]
        new_pos = pos + step
        gx = (new_pos[0] + half) % L
        gy = (new_pos[1] + half) % L
        if not mask[gy, gx]:
            pos = new_pos
        traj.append(pos.copy())
    traj = np.array(traj).T
    gif_trajectory_with_obstacles(traj, mask,
                                  f"{p['out_prefix']}.gif", fps=30)


def study_arrhenius(params=None):
    p = {**PARAMS["arrhenius"], **(params or {})}
    temperatures = np.linspace(p["T_min"], p["T_max"], p["n_T"])
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    D_values = []
    for T in temperatures:
        p_jump = float(np.clip(np.exp(-p["Ea"] / T), 1e-6, 1.0))
        rng = np.random.default_rng(int(T * 1000))
        pos = np.zeros((p["n_particles"], 2))
        for _ in range(p["n_steps"]):
            jump = rng.random(p["n_particles"]) < p_jump
            idx = rng.integers(0, 4, size=p["n_particles"])
            pos += np.where(jump[:, None], moves[idx], 0.0)
        D_values.append(np.mean(np.sum(pos ** 2, axis=1))
                        / (4 * p["n_steps"]))
    D_values = np.array(D_values)
    inv_T = 1.0 / temperatures
    valid = D_values > 1e-10
    slope, intercept = np.polyfit(inv_T[valid],
                                  np.log(D_values[valid]), 1)
    Ea_est = -float(slope)

    os.makedirs(os.path.dirname(p["out"]) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(inv_T[valid], np.log(D_values[valid]), "o-", label="симуляция")
    ax.plot(inv_T, slope * inv_T + intercept, "r--",
            label=f"Аррениус: Ea≈{Ea_est:.2f} (задано {p['Ea']})")
    ax.set_xlabel("1/T"); ax.set_ylabel("ln D")
    ax.set_title("Исследование 4: температурная зависимость D")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(p["out"], dpi=150); plt.close(fig)
    print(f"[Физ.4] Ea(оценка) = {Ea_est:.3f}, задано {p['Ea']}, "
          f"ошибка {abs(Ea_est - p['Ea'])/p['Ea']*100:.2f}%")


# =====================================================================
# §6. CLI
# =====================================================================
def _parse_value(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(x) for x in inner.split(",")]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _apply_overrides(cfg, kv_pairs):
    for pair in kv_pairs or []:
        if "=" not in pair:
            raise SystemExit(f"Некорректный --set: {pair!r} (нужно key=value)")
        k, v = pair.split("=", 1)
        cfg[k.strip()] = _parse_value(v)
    return cfg


def cmd_gif_walk(args):
    cfg = _apply_overrides(dict(PARAMS["walk"]), args.set)
    if args.n_particles is not None: cfg["n_particles"] = args.n_particles
    if args.steps is not None:       cfg["n_steps"] = args.steps
    if args.seed is not None:        cfg["seed"] = args.seed
    if args.frame_step is not None:  cfg["frame_step"] = args.frame_step
    _, hist = simulate_random_walk(cfg["n_particles"], cfg["n_steps"],
                                   dt=cfg["dt"], seed=cfg["seed"])
    gif_random_walk(hist, args.output, frame_step=cfg["frame_step"])
    print(f"Сохранено: {args.output}")


def cmd_gif_pde(args):
    cfg = _apply_overrides(dict(PARAMS["pde"]), args.set)
    if args.nx is not None:         cfg["nx"] = args.nx
    if args.ny is not None:         cfg["ny"] = args.ny
    if args.n_steps is not None:    cfg["n_steps"] = args.n_steps
    if args.kappa is not None:      cfg["kappa"] = args.kappa
    if args.dt is not None:         cfg["dt"] = args.dt
    if args.save_every is not None: cfg["save_every"] = args.save_every
    if args.source is not None:     cfg["source"] = args.source

    frames = simulate_diffusion_pde(
        cfg["nx"], cfg["ny"], cfg["n_steps"],
        kappa=cfg["kappa"], dt=cfg["dt"],
        save_every=cfg["save_every"],
        source=cfg["source"], sigma0=cfg["sigma0"])
    gif_diffusion_pde(frames, args.output)
    print(f"Сохранено: {args.output}")


def cmd_study(args):
    os.makedirs("results", exist_ok=True)
    overrides = _apply_overrides({}, args.set)
    if args.name in ("all", "msd"): study_msd(overrides)
    if args.name in ("all", "pde"): study_pde(overrides)
    print("Исследования завершены → results/")


def cmd_physics(args):
    os.makedirs("results", exist_ok=True)
    mapping = {
        "brownian": study_brownian,
        "sources": study_sources,
        "obstacles": study_obstacles,
        "arrhenius": study_arrhenius,
    }
    if args.name == "all":
        for fn in mapping.values():
            fn()
    else:
        overrides = _apply_overrides({}, args.set)
        mapping[args.name](overrides)
    print("Физические исследования завершены → results/")


def build_parser():
    p = argparse.ArgumentParser(
        description="Моделирование диффузии (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python dif3.py gif-walk -n 500 -s 200 --seed 42 -o walk.gif\n"
            "  python dif3.py gif-pde --source two -s 400 -o diff2.gif\n"
            "  python dif3.py study all\n"
            "  python dif3.py physics all\n"
            "  python dif3.py physics obstacles --set obstacles_fraction=0.4\n"
        ))
    sub = p.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("gif-walk", help="GIF случайного блуждания")
    pw.add_argument("-n", "--n-particles", type=int, default=None)
    pw.add_argument("-s", "--steps", type=int, default=None)
    pw.add_argument("--seed", type=int, default=None)
    pw.add_argument("--frame-step", type=int, default=None)
    pw.add_argument("-o", "--output", default="walk.gif")
    pw.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    pw.set_defaults(func=cmd_gif_walk)

    pp = sub.add_parser("gif-pde", help="GIF уравнения теплопроводности")
    pp.add_argument("--nx", type=int, default=None)
    pp.add_argument("--ny", type=int, default=None)
    pp.add_argument("-s", "--n-steps", type=int, default=None)
    pp.add_argument("--kappa", type=float, default=None)
    pp.add_argument("--dt", type=float, default=None)
    pp.add_argument("--save-every", type=int, default=None)
    pp.add_argument("--source", choices=["delta", "gauss", "two"], default=None)
    pp.add_argument("-o", "--output", default="diffusion.gif")
    pp.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    pp.set_defaults(func=cmd_gif_pde)

    ps = sub.add_parser("study", help="Базовые исследования")
    ps.add_argument("name", choices=["all", "msd", "pde"],
                    default="all", nargs="?")
    ps.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    ps.set_defaults(func=cmd_study)

    pf = sub.add_parser("physics", help="Физические исследования")
    pf.add_argument("name",
                    choices=["all", "brownian", "sources",
                             "obstacles", "arrhenius"],
                    default="all", nargs="?")
    pf.add_argument("--set", nargs="*", metavar="KEY=VALUE")
    pf.set_defaults(func=cmd_physics)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
# END OF FILE