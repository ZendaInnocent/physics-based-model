# Physics-Based Model

A Python project for simulating physics-based models, specifically focusing on nozzle clogging phenomena.

## Setup

### Prerequisites

- Python 3.13 or higher
- UV (recommended) or pip for dependency management

### Installation

1. **Using UV (recommended):**

   ```bash
   uv sync
   ```

2. **Using pip:**

   ```bash
   pip install -e .
   ```

### Running the Project

After installation, run the main simulation:

```bash
python -m src.main
```

Or directly:

```bash
python src/main.py
```

### Project Structure

- `src/main.py` - Main entry point
- `src/nozzle_clogging/` - Core physics simulation modules
- `src/notebooks/` - Analysis and visualization notebooks
- `src/tests/` - Unit tests

### Running Notebooks

The project includes marimo notebooks for analysis and visualizations:

1. **Analysis notebook** - Sensitivity and convergence analysis

   ```bash
   marimo edit src/notebooks/analysis.py
   ```

2. **Visualizations notebook** - Publication-quality figures
   ```bash
   marimo edit src/notebooks/visualizations.py
   ```

To run notebooks headlessly (without UI):

```bash
marimo run src/notebooks/analysis.py
marimo run src/notebooks/visualizations.py
```
