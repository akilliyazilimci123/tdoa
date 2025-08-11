import argparse
import csv
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# Physical constants
C_VACUUM_M_PER_NS = 0.299_792_458  # m/ns
REFRACTIVE_INDEX_AIR = 1.000_293
C_AIR_M_PER_NS = C_VACUUM_M_PER_NS / REFRACTIVE_INDEX_AIR

# WGS84 constants
WGS84_A = 6378137.0  # semi-major axis in meters
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


@dataclass
class Receiver:
    receiver_id: int
    lat_deg: float
    lon_deg: float
    alt_m: float
    fiber_delay_ns: float

    def to_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m)


@dataclass
class Target:
    target_id: int
    lat_deg: float
    lon_deg: float
    alt_m: float

    def to_ecef(self) -> np.ndarray:
        return geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m)


# ---------- Geodesy ----------

def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - WGS84_E2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / max(1e-12, math.cos(lat)) - N
        lat = math.atan2(z, p * (1 - WGS84_E2 * (N / (N + alt))))
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(1e-12, math.cos(lat)) - N
    return (math.degrees(lat), math.degrees(lon), alt)


def enu_rotation_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat = math.radians(lat0_deg)
    lon = math.radians(lon0_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    e = np.array([-sin_lon, cos_lon, 0.0])
    n = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    u = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    R = np.vstack([e, n, u])
    return R


def ecef_to_enu(xyz: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    ref = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    R = enu_rotation_matrix(ref_lat_deg, ref_lon_deg)
    return R @ (xyz - ref)


def enu_to_ecef(enu: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    ref = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    R = enu_rotation_matrix(ref_lat_deg, ref_lon_deg)
    return ref + R.T @ enu


# ---------- IO helpers ----------

def load_receivers(path: str) -> List[Receiver]:
    receivers: List[Receiver] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            receivers.append(
                Receiver(
                    receiver_id=idx,
                    lat_deg=float(row["lat"]),
                    lon_deg=float(row["lon"]),
                    alt_m=float(row["alt_m"]),
                    fiber_delay_ns=float(row["fiber_delay_ns"]),
                )
            )
    return receivers


def save_targets(path: str, targets: List[Target]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lat", "lon", "alt_m"])
        for t in targets:
            writer.writerow([f"{t.lat_deg:.7f}", f"{t.lon_deg:.7f}", f"{t.alt_m:.3f}"])


def load_targets(path: str) -> List[Target]:
    targets: List[Target] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            targets.append(
                Target(
                    target_id=idx,
                    lat_deg=float(row["lat"]),
                    lon_deg=float(row["lon"]),
                    alt_m=float(row["alt_m"]),
                )
            )
    return targets


def save_arrivals(path: str, arrivals: List[Tuple[int, int, float]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "receiver_id", "arrival_time_ns"])
        for tgt_id, rx_id, t_ns in arrivals:
            writer.writerow([tgt_id, rx_id, f"{t_ns:.8f}"])


def load_arrivals(path: str) -> List[Tuple[int, int, float]]:
    out: List[Tuple[int, int, float]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((int(row["target_id"]), int(row["receiver_id"]), float(row["arrival_time_ns"])) )
    return out


# ---------- Synthetic data ----------

def generate_targets_near_receivers(
    receivers: List[Receiver],
    num_targets: int,
    max_radius_m: float,
    seed: int = 42,
) -> List[Target]:
    rng = np.random.default_rng(seed)
    mean_lat = sum(r.lat_deg for r in receivers) / len(receivers)
    mean_lon = sum(r.lon_deg for r in receivers) / len(receivers)
    mean_alt = sum(r.alt_m for r in receivers) / len(receivers)

    targets: List[Target] = []
    for i in range(num_targets):
        radius = max_radius_m * math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()
        de = radius * math.cos(theta)
        dn = radius * math.sin(theta)
        du = rng.uniform(-50.0, 50.0)
        # meters per deg approximate
        meters_per_deg_lat = 111_132.92 - 559.82 * math.cos(2 * math.radians(mean_lat)) + 1.175 * math.cos(4 * math.radians(mean_lat))
        meters_per_deg_lon = 111_412.84 * math.cos(math.radians(mean_lat)) - 93.5 * math.cos(3 * math.radians(mean_lat))
        dlat_deg = dn / meters_per_deg_lat
        dlon_deg = de / meters_per_deg_lon
        lat = mean_lat + dlat_deg
        lon = mean_lon + dlon_deg
        alt = mean_alt + du
        targets.append(Target(target_id=i, lat_deg=lat, lon_deg=lon, alt_m=alt))
    return targets


def compute_arrivals(
    receivers: List[Receiver],
    targets: List[Target],
    c_m_per_ns: float = C_AIR_M_PER_NS,
) -> List[Tuple[int, int, float]]:
    arrivals: List[Tuple[int, int, float]] = []
    rx_ecef = [r.to_ecef() for r in receivers]
    for t in targets:
        t_ecef = t.to_ecef()
        for r, r_ecef in zip(receivers, rx_ecef):
            d_m = float(np.linalg.norm(t_ecef - r_ecef))
            tof_ns = d_m / c_m_per_ns
            arrival_ns = tof_ns + r.fiber_delay_ns
            arrivals.append((t.target_id, r.receiver_id, arrival_ns))
    return arrivals


# ---------- Estimation algorithms ----------

def estimate_target_position_from_tdoa(
    receivers: List[Receiver],
    arrival_times_ns: List[float],
    c_m_per_ns: float = C_AIR_M_PER_NS,
    max_iter: int = 100,
    tol_update_m: float = 1e-4,
) -> np.ndarray:
    assert len(receivers) == len(arrival_times_ns) >= 3
    # Work in local ENU coordinates around centroid for better conditioning
    lat0 = sum(r.lat_deg for r in receivers) / len(receivers)
    lon0 = sum(r.lon_deg for r in receivers) / len(receivers)
    alt0 = sum(r.alt_m for r in receivers) / len(receivers)
    rx_ecef = np.stack([r.to_ecef() for r in receivers], axis=0)
    rx_enu = np.stack([ecef_to_enu(p, lat0, lon0, alt0) for p in rx_ecef], axis=0)

    fiber = np.array([r.fiber_delay_ns for r in receivers])
    t = np.array(arrival_times_ns)
    t_corr = t - fiber
    ref = 0
    delta_t = t_corr - t_corr[ref]

    # Initial guess: centroid in ENU
    x = rx_enu.mean(axis=0).copy()

    lam = 1e-2  # LM damping
    prev_cost = np.inf

    for _ in range(max_iter):
        d = np.linalg.norm(x - rx_enu, axis=1)
        d = np.maximum(d, 1e-6)
        r_list = []
        J_list = []
        for i in range(1, len(receivers)):
            ri = d[i] - d[ref] - c_m_per_ns * (delta_t[i] - delta_t[ref])
            r_list.append(ri)
            ui = (x - rx_enu[i]) / d[i]
            u0 = (x - rx_enu[ref]) / d[ref]
            Ji = ui - u0
            J_list.append(Ji)
        r_vec = np.array(r_list)
        J = np.vstack(J_list)
        cost = float(r_vec @ r_vec)
        H = J.T @ J
        g = J.T @ r_vec
        # LM step
        H_lm = H + lam * np.eye(3)
        try:
            dx = -np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            dx = -np.linalg.pinv(H_lm) @ g
        x_new = x + dx
        # Evaluate new cost
        d_new = np.linalg.norm(x_new - rx_enu, axis=1)
        d_new = np.maximum(d_new, 1e-6)
        r_new = []
        for i in range(1, len(receivers)):
            rni = d_new[i] - d_new[ref] - c_m_per_ns * (delta_t[i] - delta_t[ref])
            r_new.append(rni)
        cost_new = float(np.dot(r_new, r_new))
        if cost_new < cost:
            x = x_new
            prev_cost = cost_new
            lam = max(lam * 0.5, 1e-6)
        else:
            lam = min(lam * 2.0, 1e6)
        if float(np.linalg.norm(dx)) < tol_update_m:
            break

    x_ecef = enu_to_ecef(x, lat0, lon0, alt0)
    return x_ecef


def batch_estimate_targets(
    receivers: List[Receiver],
    arrivals: List[Tuple[int, int, float]],
    c_m_per_ns: float = C_AIR_M_PER_NS,
) -> List[Target]:
    num_receivers = len(receivers)
    by_target: dict[int, List[Tuple[int, float]]] = {}
    for tgt_id, rx_id, t_ns in arrivals:
        by_target.setdefault(tgt_id, []).append((rx_id, t_ns))
    est_targets: List[Target] = []
    for tgt_id in sorted(by_target.keys()):
        rows = sorted(by_target[tgt_id], key=lambda x: x[0])
        if len(rows) != num_receivers:
            raise ValueError(f"Target {tgt_id} has {len(rows)} arrivals, expected {num_receivers}")
        times = [t for _, t in rows]
        x_ecef = estimate_target_position_from_tdoa(receivers, times, c_m_per_ns=c_m_per_ns)
        lat, lon, alt = ecef_to_geodetic(x_ecef)
        est_targets.append(Target(target_id=tgt_id, lat_deg=lat, lon_deg=lon, alt_m=alt))
    return est_targets


# ---------- Calibration (Problem 2) ----------

def calibrate_fiber_delays_relative(
    receivers: List[Receiver],
    arrivals: List[Tuple[int, int, float]],
    true_targets: List[Target],
    c_m_per_ns: float = C_AIR_M_PER_NS,
) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (fiber_est_ns, corrections_ns) where fiber_est anchored at receiver 0 equals its input value
    num_receivers = len(receivers)
    rx_pos = np.stack([r.to_ecef() for r in receivers], axis=0)
    fiber_in = np.array([r.fiber_delay_ns for r in receivers])

    # Group arrivals by target
    by_target: dict[int, List[Tuple[int, float]]] = {}
    for tgt_id, rx_id, t_ns in arrivals:
        by_target.setdefault(tgt_id, []).append((rx_id, t_ns))

    # Accumulate estimates for (f_i - f_0)
    diffs_accum: List[List[float]] = [[] for _ in range(num_receivers)]
    for tgt in true_targets:
        t_ecef = tgt.to_ecef()
        rows = sorted(by_target[tgt.target_id], key=lambda x: x[0])
        times = np.array([t for _, t in rows])
        ref = 0
        for i in range(1, num_receivers):
            d_i = float(np.linalg.norm(t_ecef - rx_pos[i]))
            d_0 = float(np.linalg.norm(t_ecef - rx_pos[ref]))
            rhs = (times[i] - times[ref]) - (d_i - d_0) / c_m_per_ns
            diffs_accum[i].append(rhs)

    f_rel = np.zeros(num_receivers)
    for i in range(1, num_receivers):
        if len(diffs_accum[i]) == 0:
            raise ValueError("Insufficient arrivals for calibration")
        f_rel[i] = float(np.mean(diffs_accum[i]))

    # Anchor absolute by keeping f0 same as input
    f_abs = np.zeros(num_receivers)
    f_abs[0] = fiber_in[0]
    for i in range(1, num_receivers):
        f_abs[i] = f_abs[0] + f_rel[i]

    corrections = f_abs - fiber_in
    return f_abs, corrections


# ---------- Joint calibration (Problem 3) ----------

def calibrate_fibers_and_one_station(
    receivers: List[Receiver],
    arrivals: List[Tuple[int, int, float]],
    true_targets: List[Target],
    unknown_station_id: int,
    c_m_per_ns: float = C_AIR_M_PER_NS,
    max_iter: int = 100,
    tol_update: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    num_receivers = len(receivers)
    rx_pos = np.stack([r.to_ecef() for r in receivers], axis=0)
    fiber_in = np.array([r.fiber_delay_ns for r in receivers])

    # Group arrivals by target
    by_target: dict[int, List[Tuple[int, float]]] = {}
    for tgt_id, rx_id, t_ns in arrivals:
        by_target.setdefault(tgt_id, []).append((rx_id, t_ns))

    # Variables: r_s (3) and delta_f[1..N-1] (receiver 0 is reference with 0 correction)
    s = unknown_station_id
    r_s = rx_pos[s].copy()
    delta_f = np.zeros(num_receivers)
    delta_f[0] = 0.0

    # Build measurement arrays for speed
    targets_sorted = sorted(true_targets, key=lambda t: t.target_id)

    for _ in range(max_iter):
        # Build residuals and Jacobian
        residuals: List[float] = []
        J_rows: List[np.ndarray] = []
        for tgt in targets_sorted:
            t_ecef = tgt.to_ecef()
            rows = sorted(by_target[tgt.target_id], key=lambda x: x[0])
            times = np.array([t for _, t in rows])
            ref = 0
            # distances given current r_s
            d = np.linalg.norm(t_ecef - rx_pos, axis=1)
            # Update any index equal to s with current r_s
            if True:
                d_s = float(np.linalg.norm(t_ecef - r_s))
                d[s] = d_s
            if ref == s:
                d_ref = float(np.linalg.norm(t_ecef - r_s))
            else:
                d_ref = d[ref]

            for i in range(1, num_receivers):
                # residual: (t_i - t_0) - ((fiber_i+df_i) - (fiber_0+df_0)) - (d_i - d_0)/c = 0
                res = (times[i] - times[ref]) - ((fiber_in[i] + delta_f[i]) - (fiber_in[ref] + delta_f[ref])) - (d[i] - d[ref]) / c_m_per_ns
                residuals.append(float(res))
                # Jacobian wrt [r_s(x,y,z), delta_f(1..N-1)]
                # Partials wrt r_s
                Jr = np.zeros(3)
                # -(1/c)*(indicator(i==s)*(r_s - x_k)/d_i - indicator(ref==s)*(r_s - x_k)/d_ref)
                if i == s:
                    if d[i] > 1e-9:
                        Jr += -(r_s - t_ecef) / (d[i] * c_m_per_ns)
                if ref == s:
                    if d_ref > 1e-9:
                        Jr += (r_s - t_ecef) / (d_ref * c_m_per_ns)
                # Partials wrt delta_f(1..N-1)
                Jdf = np.zeros(num_receivers - 1)
                # df index mapping: j in [1..N-1] maps to position j-1
                if i >= 1:
                    Jdf[i - 1] = -1.0
                # Stack row
                J_row = np.concatenate([Jr, Jdf])
                J_rows.append(J_row)
        r_vec = np.array(residuals)
        J = np.vstack(J_rows)
        H = J.T @ J
        g = J.T @ r_vec
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(H) @ g
        # Update
        r_s += step[0:3]
        delta_f[1:] += step[3:]
        if float(np.linalg.norm(step)) < tol_update:
            break

    fiber_est = fiber_in.copy()
    fiber_est[1:] = fiber_in[1:] + delta_f[1:]
    # Keep reference 0 unchanged to set gauge
    return fiber_est, r_s


# ---------- Extra utilities ----------

def meters_between_llh(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ea = geodetic_to_ecef(a[0], a[1], a[2])
    eb = geodetic_to_ecef(b[0], b[1], b[2])
    return float(np.linalg.norm(ea - eb))


def clone_receivers_with_fibers(receivers: List[Receiver], fibers_ns: np.ndarray) -> List[Receiver]:
    out: List[Receiver] = []
    for r, f in zip(receivers, fibers_ns):
        out.append(Receiver(r.receiver_id, r.lat_deg, r.lon_deg, r.alt_m, float(f)))
    return out


def clone_receivers_with_station(receivers: List[Receiver], s_id: int, new_llh: Tuple[float, float, float]) -> List[Receiver]:
    out: List[Receiver] = []
    for r in receivers:
        if r.receiver_id == s_id:
            out.append(Receiver(r.receiver_id, new_llh[0], new_llh[1], new_llh[2], r.fiber_delay_ns))
        else:
            out.append(Receiver(r.receiver_id, r.lat_deg, r.lon_deg, r.alt_m, r.fiber_delay_ns))
    return out


def cmd_apply_fiber_calibration(args):
    receivers = load_receivers(args.receivers)
    # Load calibration map receiver_id -> fiber_delay_est_ns
    calib: dict[int, float] = {}
    with open(args.calibration_csv, newline="") as f:
        reader = csv.DictReader(f)
        rid_key = "receiver_id"
        fest_key = "fiber_delay_est_ns"
        for row in reader:
            rid = int(row[rid_key])
            fest = float(row[fest_key])
            calib[rid] = fest
    # Write new receivers CSV
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "alt_m", "fiber_delay_ns"])
        for r in receivers:
            new_f = calib.get(r.receiver_id, r.fiber_delay_ns)
            w.writerow([f"{r.lat_deg:.7f}", f"{r.lon_deg:.7f}", f"{r.alt_m:.3f}", f"{new_f:.6f}"])


def cmd_update_station_from_csv(args):
    receivers = load_receivers(args.receivers)
    # Load station update
    with open(args.station_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) != 1:
            raise ValueError("station_csv must contain exactly one row")
        row = rows[0]
        s_id = int(row["receiver_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        alt = float(row["alt_m"])
    # Write new receivers CSV
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "alt_m", "fiber_delay_ns"])
        for r in receivers:
            if r.receiver_id == s_id:
                w.writerow([f"{lat:.7f}", f"{lon:.7f}", f"{alt:.3f}", f"{r.fiber_delay_ns:.6f}"])
            else:
                w.writerow([f"{r.lat_deg:.7f}", f"{r.lon_deg:.7f}", f"{r.alt_m:.3f}", f"{r.fiber_delay_ns:.6f}"])


# ---------- CLI ----------

def cmd_generate_targets(args):
    receivers = load_receivers(args.receivers)
    targets = generate_targets_near_receivers(receivers, args.num_targets, args.max_radius_m, seed=args.seed)
    save_targets(args.output, targets)


def cmd_compute_arrivals(args):
    receivers = load_receivers(args.receivers)
    targets = load_targets(args.targets)
    arrivals = compute_arrivals(receivers, targets, c_m_per_ns=C_AIR_M_PER_NS)
    save_arrivals(args.output, arrivals)


def cmd_estimate_targets(args):
    receivers = load_receivers(args.receivers)
    arrivals = load_arrivals(args.arrivals)
    est_targets = batch_estimate_targets(receivers, arrivals, c_m_per_ns=C_AIR_M_PER_NS)
    # Write
    out_rows = [(t.lat_deg, t.lon_deg, t.alt_m) for t in est_targets]
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "alt_m"])
        for lat, lon, alt in out_rows:
            w.writerow([f"{lat:.7f}", f"{lon:.7f}", f"{alt:.3f}"])


def cmd_calibrate_fibers(args):
    receivers = load_receivers(args.receivers)
    arrivals = load_arrivals(args.arrivals)
    true_targets = load_targets(args.targets_true)
    fiber_est, corrections = calibrate_fiber_delays_relative(receivers, arrivals, true_targets, c_m_per_ns=C_AIR_M_PER_NS)
    # Write report CSV
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receiver_id", "fiber_delay_est_ns", "correction_ns"])
        for i, (fest, corr) in enumerate(zip(fiber_est, corrections)):
            w.writerow([i, f"{fest:.6f}", f"{corr:.6f}"])


def cmd_calibrate_fibers_and_station(args):
    receivers = load_receivers(args.receivers)
    arrivals = load_arrivals(args.arrivals)
    true_targets = load_targets(args.targets_true)
    fiber_est, r_s_est = calibrate_fibers_and_one_station(
        receivers, arrivals, true_targets, unknown_station_id=args.unknown_station_id, c_m_per_ns=C_AIR_M_PER_NS
    )
    lat_s, lon_s, alt_s = ecef_to_geodetic(r_s_est)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receiver_id", "fiber_delay_est_ns"])
        for i, fest in enumerate(fiber_est):
            w.writerow([i, f"{fest:.6f}"])
    with open(args.output_station, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["receiver_id", "lat", "lon", "alt_m"])
        w.writerow([args.unknown_station_id, f"{lat_s:.7f}", f"{lon_s:.7f}", f"{alt_s:.3f}"])


def cmd_self_test(args):
    rng = np.random.default_rng(args.seed)
    true_receivers = load_receivers(args.receivers)
    num_rx = len(true_receivers)
    true_fibers = np.array([r.fiber_delay_ns for r in true_receivers])

    trials = args.num_trials
    targ_per_trial = args.targets_per_trial

    # Stats accumulators
    pos_rmse_list = []
    fiber_rel_rmse_list = []
    joint_pos_rmse_list = []
    joint_fiber_rel_rmse_list = []
    station_err_m_list = []

    for trial in range(trials):
        # Generate targets
        targets = generate_targets_near_receivers(true_receivers, targ_per_trial, args.max_radius_m, seed=int(rng.integers(0, 1_000_000)))
        # Compute arrivals with true receivers and true fibers
        arrivals = compute_arrivals(true_receivers, targets, c_m_per_ns=C_AIR_M_PER_NS)

        # Create wrong fibers
        if args.fiber_noise_ns_std > 0:
            fiber_noise = rng.normal(0.0, args.fiber_noise_ns_std, size=num_rx)
        else:
            fiber_noise = np.zeros(num_rx)
        wrong_fibers = true_fibers + fiber_noise
        wrong_receivers = clone_receivers_with_fibers(true_receivers, wrong_fibers)

        # Fiber-only calibration using true targets
        f_est, _ = calibrate_fiber_delays_relative(wrong_receivers, arrivals, targets, c_m_per_ns=C_AIR_M_PER_NS)
        # Evaluate relative fiber error
        rel_true = true_fibers - true_fibers[0]
        rel_est = f_est - f_est[0]
        rel_err = rel_est - rel_true
        fiber_rel_rmse = float(np.sqrt(np.mean(rel_err[1:] ** 2)))  # exclude ref 0
        fiber_rel_rmse_list.append(fiber_rel_rmse)

        # Apply estimated fibers and re-estimate positions; TDOA should be insensitive to absolute offset
        rec_cal = clone_receivers_with_fibers(true_receivers, f_est)
        # Use arrivals from true setup; estimate target positions with rec_cal (which differs only by fiber gauge)
        est_targets = batch_estimate_targets(rec_cal, arrivals, c_m_per_ns=C_AIR_M_PER_NS)
        # Compute position RMSE in meters
        errs = []
        for t_true, t_est in zip(targets, est_targets):
            e = meters_between_llh((t_true.lat_deg, t_true.lon_deg, t_true.alt_m), (t_est.lat_deg, t_est.lon_deg, t_est.alt_m))
            errs.append(e)
        pos_rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0
        pos_rmse_list.append(pos_rmse)

        if args.station_noise_m_std > 0:
            # Pick station id
            s_id = args.unknown_station_id
            # Perturb station position in ENU around its own location
            s = true_receivers[s_id]
            enu_noise = rng.normal(0.0, args.station_noise_m_std, size=3)
            s_ecef = s.to_ecef()
            lat0, lon0, alt0 = s.lat_deg, s.lon_deg, s.alt_m
            s_enu = np.array([0.0, 0.0, 0.0])
            s_ecef_pert = enu_to_ecef(enu_noise, lat0, lon0, alt0)
            # Convert back to geodetic for wrong receiver list
            s_lat, s_lon, s_alt = ecef_to_geodetic(s_ecef_pert)
            wrong_receivers_joint = clone_receivers_with_station(wrong_receivers, s_id, (s_lat, s_lon, s_alt))
            # Joint calibration
            f_joint_est, r_s_est = calibrate_fibers_and_one_station(wrong_receivers_joint, arrivals, targets, unknown_station_id=s_id, c_m_per_ns=C_AIR_M_PER_NS)
            rel_joint = f_joint_est - f_joint_est[0]
            rel_err_joint = rel_joint - rel_true
            joint_fiber_rel_rmse = float(np.sqrt(np.mean(rel_err_joint[1:] ** 2)))
            joint_fiber_rel_rmse_list.append(joint_fiber_rel_rmse)
            # Station error
            s_true_llh = (s.lat_deg, s.lon_deg, s.alt_m)
            s_est_llh = ecef_to_geodetic(r_s_est)
            station_err_m = meters_between_llh(s_true_llh, s_est_llh)
            station_err_m_list.append(station_err_m)
            # After joint calib, update receivers and re-estimate positions
            rec_joint_cal = clone_receivers_with_fibers(true_receivers, f_joint_est)
            est_targets_joint = batch_estimate_targets(rec_joint_cal, arrivals, c_m_per_ns=C_AIR_M_PER_NS)
            errs_joint = []
            for t_true, t_est in zip(targets, est_targets_joint):
                e = meters_between_llh((t_true.lat_deg, t_true.lon_deg, t_true.alt_m), (t_est.lat_deg, t_est.lon_deg, t_est.alt_m))
                errs_joint.append(e)
            joint_pos_rmse = float(np.sqrt(np.mean(np.square(errs_joint)))) if errs_joint else 0.0
            joint_pos_rmse_list.append(joint_pos_rmse)

    def summarize(arr):
        if not arr:
            return (None, None, None)
        a = np.array(arr)
        return (float(np.mean(a)), float(np.median(a)), float(np.percentile(a, 95)))

    fiber_mean, fiber_med, fiber_p95 = summarize(fiber_rel_rmse_list)
    pos_mean, pos_med, pos_p95 = summarize(pos_rmse_list)
    joint_fiber_mean, joint_fiber_med, joint_fiber_p95 = summarize(joint_fiber_rel_rmse_list)
    joint_pos_mean, joint_pos_med, joint_pos_p95 = summarize(joint_pos_rmse_list)
    st_mean, st_med, st_p95 = summarize(station_err_m_list)

    # Print concise report
    print("Fiber-only calibration (relative, ns): RMSE mean/median/p95 = ", fiber_mean, fiber_med, fiber_p95)
    print("Post-calibration position RMSE (m):    mean/median/p95 = ", pos_mean, pos_med, pos_p95)
    if station_err_m_list:
        print("Joint calib fiber (relative, ns):    RMSE mean/median/p95 = ", joint_fiber_mean, joint_fiber_med, joint_fiber_p95)
        print("Joint calib station error (m):       mean/median/p95 = ", st_mean, st_med, st_p95)
        print("Joint post-calib pos RMSE (m):       mean/median/p95 = ", joint_pos_mean, joint_pos_med, joint_pos_p95)

    # Optional CSV report
    if args.report:
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "median", "p95"])
            w.writerow(["fiber_rel_rmse_ns", fiber_mean, fiber_med, fiber_p95])
            w.writerow(["pos_rmse_m", pos_mean, pos_med, pos_p95])
            if station_err_m_list:
                w.writerow(["joint_fiber_rel_rmse_ns", joint_fiber_mean, joint_fiber_med, joint_fiber_p95])
                w.writerow(["joint_station_err_m", st_mean, st_med, st_p95])
                w.writerow(["joint_pos_rmse_m", joint_pos_mean, joint_pos_med, joint_pos_p95])


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TDOA solver and calibration tool")
    sub = p.add_subparsers(required=True)

    g = sub.add_parser("generate-targets", help="Generate synthetic targets near receivers")
    g.add_argument("--receivers", required=True)
    g.add_argument("--num-targets", type=int, default=4)
    g.add_argument("--max-radius-m", type=float, default=10_000.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", required=True)
    g.set_defaults(func=cmd_generate_targets)

    a = sub.add_parser("compute-arrivals", help="Compute arrival times for targets and receivers")
    a.add_argument("--receivers", required=True)
    a.add_argument("--targets", required=True)
    a.add_argument("--output", required=True)
    a.set_defaults(func=cmd_compute_arrivals)

    e = sub.add_parser("estimate-targets", help="Estimate target positions from arrivals and receivers")
    e.add_argument("--receivers", required=True)
    e.add_argument("--arrivals", required=True)
    e.add_argument("--output", required=True)
    e.set_defaults(func=cmd_estimate_targets)

    cf = sub.add_parser("calibrate-fibers", help="Calibrate fiber delays (relative, anchored to rx0)")
    cf.add_argument("--receivers", required=True)
    cf.add_argument("--arrivals", required=True)
    cf.add_argument("--targets-true", required=True)
    cf.add_argument("--output", required=True)
    cf.set_defaults(func=cmd_calibrate_fibers)

    cs = sub.add_parser("calibrate-fibers-and-station", help="Calibrate fiber delays and one station position")
    cs.add_argument("--receivers", required=True)
    cs.add_argument("--arrivals", required=True)
    cs.add_argument("--targets-true", required=True)
    cs.add_argument("--unknown-station-id", type=int, required=True)
    cs.add_argument("--output", required=True, help="Output CSV for estimated fiber delays")
    cs.add_argument("--output-station", required=True, help="Output CSV for estimated station position")
    cs.set_defaults(func=cmd_calibrate_fibers_and_station)

    af = sub.add_parser("apply-fiber-calibration", help="Write new receivers CSV with calibrated fiber delays")
    af.add_argument("--receivers", required=True)
    af.add_argument("--calibration-csv", required=True)
    af.add_argument("--output", required=True)
    af.set_defaults(func=cmd_apply_fiber_calibration)

    us = sub.add_parser("update-station-from-csv", help="Write new receivers CSV with one station lat/lon/alt updated")
    us.add_argument("--receivers", required=True)
    us.add_argument("--station-csv", required=True)
    us.add_argument("--output", required=True)
    us.set_defaults(func=cmd_update_station_from_csv)

    st = sub.add_parser("self-test", help="Monte Carlo self-test for calibration and solver")
    st.add_argument("--receivers", required=True)
    st.add_argument("--num-trials", type=int, default=100)
    st.add_argument("--targets-per-trial", type=int, default=4)
    st.add_argument("--max-radius-m", type=float, default=10_000.0)
    st.add_argument("--fiber-noise-ns-std", type=float, default=50.0)
    st.add_argument("--station-noise-m-std", type=float, default=10.0)
    st.add_argument("--unknown-station-id", type=int, default=2)
    st.add_argument("--seed", type=int, default=123)
    st.add_argument("--report", default="")
    st.set_defaults(func=cmd_self_test)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()