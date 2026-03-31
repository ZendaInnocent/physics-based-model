from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "exports" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PARTICLE_SIZE_RANGES = {
    "Fine": (10, 50),
    "Medium": (50, 150),
    "Coarse": (150, 300),
}

PARAM_RANGES = {
    "TSS": (10, 500),
    "pressure": (100, 400),
    "nozzle_diameter": (1.5, 6),
    "duration": (0.5, 24),
}

RHO_WATER = 1000
RHO_SEDIMENT = 2650
g = 9.81
NU = 1e-6
CD = 0.85
LOGISTIC_SCALE = 1.0

DP_DN_OBSTRUCTION_THRESHOLD = 0.05
DP_DN_RISK_THRESHOLD = 0.14
STOKES_CRITICAL = 0.1
SETTLING_VELOCITY_RATIO_CRITICAL = 0.1
VELOCITY_SHEAR_THRESHOLD = 8.0

RANDOM_SEED = 42

RISK_LOW_THRESHOLD = 0.30
RISK_MODERATE_THRESHOLD = 0.50
RISK_LEVELS = ["Low", "Moderate", "High"]
