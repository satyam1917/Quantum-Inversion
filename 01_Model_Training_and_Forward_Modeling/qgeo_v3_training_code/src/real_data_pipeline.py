# =============================================================================
# real_data_pipeline.py
# Preprocesses real geophysical survey data for inversion.
#
# Handles:
#   - CSV ingestion (flexible column names)
#   - Unit conversion (mGal, gamma, ppm, etc.)
#   - Bouguer correction (free-air + terrain) if raw gravity provided
#   - Regional field removal (polynomial trend removal)
#   - Radiometric normalisation and despiking
#   - Output: same format as run_complex_forward() synthetic data
#
# EXPECTED CSV FORMAT (gravity):
#   Any of: x, easting, distance, dist, x_m
#   Any of: gz, g_z, gravity, bouguer, bouguer_anomaly, free_air
#   Optional: elevation, z, topo, height
#
# EXPECTED CSV FORMAT (radiometric):
#   Same x column as above
#   Any of: k, potassium, k_pct, k_percent
#   Any of: u, uranium, eu, eu_ppm, thorium_equiv
#   Any of: th, thorium, eth, eth_ppm
# =============================================================================

import numpy as np
import os, sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Column name aliases
_X_ALIASES   = ['x', 'easting', 'distance', 'dist', 'x_m', 'utm_e', 'lon', 'longitude']
_GZ_ALIASES  = ['gz', 'g_z', 'gravity', 'bouguer', 'bouguer_anomaly',
                 'free_air', 'freeair', 'complete_bouguer', 'delta_g']
_EL_ALIASES  = ['elevation', 'z', 'topo', 'height', 'altitude', 'dem', 'elev_m']
_K_ALIASES   = ['k', 'potassium', 'k_pct', 'k_percent', 'k_%']
_U_ALIASES   = ['u', 'uranium', 'eu', 'eu_ppm', 'u_ppm']
_TH_ALIASES  = ['th', 'thorium', 'eth', 'eth_ppm', 'th_ppm']


def _find_col(header, aliases):
    """Case-insensitive column lookup."""
    hlow = [h.strip().lower() for h in header]
    for a in aliases:
        if a.lower() in hlow:
            return hlow.index(a.lower())
    return None


def load_csv(filepath):
    """
    Load a CSV file and return header + data as numpy array.
    Auto-detects delimiter (comma, tab, space).
    """
    with open(filepath, 'r') as f:
        first = f.readline().strip()
    delim = ',' if ',' in first else ('\t' if '\t' in first else None)
    data  = np.genfromtxt(filepath, delimiter=delim, names=True, dtype=float,
                           encoding='utf-8', invalid_raise=False)
    return data


def bouguer_correction(elevation_m, gravity_raw_mgal,
                        rho_bouguer=2670.0, lat_deg=22.3):
    """
    Apply simple complete Bouguer correction to raw gravity data.

    B = g_raw - g_normal(lat) + FAC - BC
      FAC (free-air correction) = 0.3086 * h  mGal/m
      BC  (Bouguer slab)        = 0.04193 * rho * h  mGal/m

    Parameters
    ----------
    elevation_m       : array (n,)  elevation in metres
    gravity_raw_mgal  : array (n,)  raw observed gravity in mGal
    rho_bouguer       : float       Bouguer density (kg/m³), default 2670
    lat_deg           : float       mean latitude of survey (degrees)

    Returns
    -------
    bouguer_anomaly : array (n,)  Bouguer anomaly in mGal
    """
    # Normal gravity at latitude (GRS80)
    lat   = np.deg2rad(lat_deg)
    g_n   = 978032.7 * (1 + 0.0053024 * np.sin(lat)**2
                          - 0.0000058 * np.sin(2*lat)**2)  # mGal

    h     = np.asarray(elevation_m, dtype=float)
    g_raw = np.asarray(gravity_raw_mgal, dtype=float)

    FAC   = 0.3086  * h                             # mGal
    BC    = 0.04193 * (rho_bouguer / 1000.0) * h    # mGal  (rho in g/cc)

    return g_raw + FAC - BC - g_n


def remove_regional(x, anomaly, degree=2):
    """
    Remove polynomial regional field from anomaly.
    Returns residual (local) anomaly.
    """
    x_n   = (x - x.mean()) / (x.max() - x.min() + 1e-8)
    coeffs = np.polyfit(x_n, anomaly, deg=degree)
    trend  = np.polyval(coeffs, x_n)
    return anomaly - trend


def despike_radio(signal, window=11, threshold=3.0):
    """
    Remove spikes from radiometric channel using median + std threshold.
    Replaces spikes with local median.
    """
    sig    = np.array(signal, dtype=float)
    median = np.median(sig)
    mad    = np.median(np.abs(sig - median)) * 1.4826   # robust std estimator
    mask   = np.abs(sig - median) > threshold * max(mad, 1e-8)
    if mask.any():
        # Replace spikes with smoothed neighbours
        smooth = savgol_filter(sig, min(window, len(sig)//2*2-1 or 3), 2)
        sig[mask] = smooth[mask]
    return sig


def load_gravity_csv(filepath, apply_bouguer=False, lat_deg=22.3,
                     remove_trend=True, trend_degree=2):
    """
    Load and preprocess gravity data from CSV.

    Parameters
    ----------
    filepath       : str   path to CSV file
    apply_bouguer  : bool  if True, apply Bouguer correction (needs elevation col)
    lat_deg        : float mean survey latitude for Bouguer correction
    remove_trend   : bool  remove polynomial regional field
    trend_degree   : int   degree of polynomial for regional removal

    Returns
    -------
    x       : (n,) observation x positions (m or relative m)
    gravity : (n,) Bouguer anomaly (mGal)
    """
    data    = load_csv(filepath)
    names   = list(data.dtype.names)

    ix  = _find_col(names, _X_ALIASES)
    igz = _find_col(names, _GZ_ALIASES)
    iel = _find_col(names, _EL_ALIASES)

    if ix is None:
        raise ValueError(f"No x/distance column found. Available: {names}")
    if igz is None:
        raise ValueError(f"No gravity column found. Available: {names}")

    x       = data[names[ix]].astype(float)
    gravity = data[names[igz]].astype(float)

    # Sort by x
    order   = np.argsort(x)
    x, gravity = x[order], gravity[order]

    # Apply Bouguer correction if raw gravity + elevation provided
    if apply_bouguer:
        if iel is None:
            raise ValueError("apply_bouguer=True but no elevation column found")
        elev    = data[names[iel]].astype(float)[order]
        gravity = bouguer_correction(elev, gravity, lat_deg=lat_deg)
        print(f"  Bouguer correction applied (lat={lat_deg}°)")

    # Remove regional trend
    if remove_trend:
        gravity = remove_regional(x, gravity, degree=trend_degree)
        print(f"  Regional trend removed (degree={trend_degree})")

    print(f"  Gravity loaded: {len(x)} points  "
          f"range {gravity.min():.2f} to {gravity.max():.2f} mGal")
    return x, gravity


def load_radiometric_csv(filepath, smooth_window=11):
    """
    Load and preprocess radiometric data from CSV.

    Returns
    -------
    x      : (n,)
    K      : (n,) % potassium
    U      : (n,) ppm uranium
    Th     : (n,) ppm thorium
    """
    data  = load_csv(filepath)
    names = list(data.dtype.names)

    ix  = _find_col(names, _X_ALIASES)
    ik  = _find_col(names, _K_ALIASES)
    iu  = _find_col(names, _U_ALIASES)
    ith = _find_col(names, _TH_ALIASES)

    if ix is None:
        raise ValueError(f"No x column found. Available: {names}")

    x = data[names[ix]].astype(float)
    order = np.argsort(x)
    x = x[order]

    def _get(idx, name):
        if idx is None:
            print(f"  [WARN] No {name} column — using zeros")
            return np.zeros(len(x))
        v = data[names[idx]].astype(float)[order]
        v = despike_radio(v)
        if smooth_window > 1:
            v = savgol_filter(v, min(smooth_window, len(v)//2*2-1 or 3), 2)
        return v

    K  = _get(ik,  'K (potassium)')
    U  = _get(iu,  'U (uranium)')
    Th = _get(ith, 'Th (thorium)')

    print(f"  Radiometric loaded: {len(x)} points  "
          f"K=[{K.min():.2f},{K.max():.2f}]  "
          f"U=[{U.min():.2f},{U.max():.2f}]  "
          f"Th=[{Th.min():.2f},{Th.max():.2f}]")
    return x, K, U, Th


def align_and_resample(x_grav, gravity, x_radio, K, U, Th, n_out=150):
    """
    Resample gravity and radiometric data onto a common uniform grid.
    Crops to the overlap region of both datasets.

    Returns
    -------
    x_out       : (n_out,)
    g_out       : (n_out,)
    radio_out   : (3, n_out)  [K, U, Th]
    """
    x_lo = max(x_grav.min(), x_radio.min())
    x_hi = min(x_grav.max(), x_radio.max())
    if x_lo >= x_hi:
        raise ValueError("Gravity and radiometric x ranges do not overlap")

    x_out = np.linspace(x_lo, x_hi, n_out)

    g_out = interp1d(x_grav, gravity,
                     kind='linear', fill_value='extrapolate')(x_out)
    K_out = interp1d(x_radio, K,  kind='linear', fill_value='extrapolate')(x_out)
    U_out = interp1d(x_radio, U,  kind='linear', fill_value='extrapolate')(x_out)
    Th_out= interp1d(x_radio, Th, kind='linear', fill_value='extrapolate')(x_out)

    radio_out = np.stack([K_out, U_out, Th_out], axis=0)
    print(f"  Aligned to {n_out} points  x=[{x_lo:.0f},{x_hi:.0f}] m")
    return x_out, g_out, radio_out


def generate_real_data_template(save_dir="real_data"):
    """
    Generate template CSV files showing expected format for real data input.
    Edit these files with your actual survey data.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Gravity template
    n = 50
    x = np.linspace(0, 10000, n)
    grav_template = np.column_stack([
        x,
        np.random.normal(0, 10, n),   # Replace with your Bouguer anomaly (mGal)
        np.random.uniform(50, 150, n), # Replace with elevation (m) — needed if raw
    ])
    np.savetxt(
        os.path.join(save_dir, "gravity_template.csv"),
        grav_template,
        delimiter=',',
        header='x,bouguer_anomaly,elevation',
        comments='',
        fmt='%.4f'
    )

    # Radiometric template
    radio_template = np.column_stack([
        x,
        np.random.uniform(0.5, 4.0, n),  # K (%)
        np.random.uniform(0.5, 5.0, n),  # eU (ppm)
        np.random.uniform(3.0, 20.0, n), # eTh (ppm)
    ])
    np.savetxt(
        os.path.join(save_dir, "radiometric_template.csv"),
        radio_template,
        delimiter=',',
        header='x,k,u,th',
        comments='',
        fmt='%.4f'
    )

    print(f"Templates saved to {save_dir}/")
    print("  gravity_template.csv  — edit with your Bouguer anomaly data")
    print("  radiometric_template.csv — edit with your K/U/Th data")
    print("  Required columns: x (metres from start of profile)")


def run_real_data_inversion(
    gravity_csv      = "real_data/gravity_template.csv",
    radio_csv        = "real_data/radiometric_template.csv",
    apply_bouguer    = False,
    lat_deg          = 22.3,
    remove_trend     = True,
    trend_degree     = 2,
    n_resample       = 150,
    results_dir      = "results",
):
    """
    Full pipeline: load -> preprocess -> inversion -> save.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inversion_engine import run_inversion

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("  REAL DATA INVERSION PIPELINE")
    print("=" * 65)

    # Load and preprocess
    print("\n[1/3] Loading gravity data...")
    x_g, g = load_gravity_csv(gravity_csv,
                               apply_bouguer=apply_bouguer,
                               lat_deg=lat_deg,
                               remove_trend=remove_trend,
                               trend_degree=trend_degree)

    print("\n[2/3] Loading radiometric data...")
    x_r, K, U, Th = load_radiometric_csv(radio_csv)

    print("\n[3/3] Aligning and running inversion...")
    x_out, g_out, radio_out = align_and_resample(x_g, g, x_r, K, U, Th, n_resample)

    best, mc_mean, mc_std, _, _ = run_inversion(
        g_out, radio_out,
        save_path=os.path.join(results_dir, "real_data_inversion.npz")
    )

    return best, mc_mean, mc_std


if __name__ == "__main__":
    generate_real_data_template()
    print("\nTo run on real data, call:")
    print("  run_real_data_inversion('your_gravity.csv', 'your_radio.csv')")
