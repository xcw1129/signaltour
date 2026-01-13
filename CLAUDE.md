# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

signaltour is a Python signal processing library designed for one-dimensional time-series oscillation data. It provides an object-oriented workflow for signal loading, preprocessing, analysis, and visualization.

## Architecture

The library follows a three-layer modular design with clear separation of concerns:

### Core Modules

1. **Signal Module** (`src/signaltour/_Signal_Module/`)
   - Core `Signal` class that encapsulates 1D time-domain signal data with sampling parameters (fs, T)
   - `Axis` hierarchy: `Axis` → `t_Axis` (time), `f_Axis` (frequency)
   - `Series` class: Generic 1D sequence data with axis information
   - Signal objects are the central data structure passed between Analysis and Plot modules
   - Submodules:
     - `core.py`: Core classes (Axis, Series, Signal, Spectra)
     - `SignalRead.py`: File/folder/dataset management (Files, Folder, Dataset)
     - `SignalSimulate.py`: Signal generation (periodic, impulse, modulation)
     - `SignalSample.py`: Resampling, padding, slicing
     - `SignalFilt.py`: Filtering (FIR, IIR, median)

2. **Analysis Module** (`src/signaltour/_Analysis_Module/`)
   - `BaseAnalysis` class defines standardized analysis workflow
   - Receives `Signal` objects, executes algorithms, returns results
   - Uses `@Analysis._plot` decorator to link analysis results with Plot module
   - Analysis classes have `isPlot` parameter to control automatic visualization
   - Submodules:
     - `core.py`: BaseAnalysis framework
     - `SpectrumAnalysis.py`: Frequency domain analysis
     - `ModeAnalysis.py`: Modal decomposition (EMD, VMD)
     - `TimeFreqAnalysis.py`: Time-frequency analysis
     - `StatsTrendAnalysis.py`: Statistical and trend analysis

3. **Plot Module** (`src/signaltour/_Plot_Module/`)
   - `BasePlot` class implements task queue-based plotting framework
   - Chain-style API: register tasks → configure → execute with `show()`
   - Plugin system via `PlotPlugin` class for extensible functionality
   - Supports multi-subplot layouts with `ncols` parameter
   - Submodules:
     - `core.py`: BasePlot and PlotPlugin framework
     - `LinePlot.py`: Time/frequency waveform plotting
     - `ImagePlot.py`: 2D spectrum visualization
     - `PlotPlugin.py`: Plugin implementations (e.g., PeakfinderPlugin)

### Module Interfaces

- **Top-level files** (`Signal.py`, `Analysis.py`, `Plot.py`): Public API entry points that re-export from internal modules
- **Internal modules** (prefixed with `_`): Implementation details, not intended for direct import
- **Assist Module** (`_Assist_Module/`): Shared utilities (Dependencies.py, Decorators.py)

### Design Philosophy

- **Signal-centric**: Signal objects carry both data and metadata (fs, T, name, unit)
- **Analysis framework**: BaseAnalysis provides template for algorithm implementation with optional visualization
- **Plot engine**: Task queue separates plot configuration from execution, enabling flexible composition

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_Signal.py

# Run with coverage
pytest --cov=signaltour
```

### Linting and Formatting
```bash
# Check code style with ruff
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Building
```bash
# Build package distributions
python -m build

# Verify build
python -m twine check dist/*
```

### Documentation
```bash
# Build Sphinx documentation (from doc/ directory)
cd doc
make html  # On Linux/Mac
make.bat html  # On Windows

# Auto-generate module documentation
python script/autogenerate_module_doc.py
```

## Code Style

- **Python version**: Requires Python ≥3.11
- **Docstring style**: NumPy convention (configured in pyproject.toml)
- **Line length**: 120 characters
- **Import sorting**: isort with signaltour as first-party
- **Type hints**: Required for public function return types (ANN201)
- **Language**: Documentation and comments are primarily in Chinese

## Key Dependencies

- numpy ≥2.0.0
- scipy ≥1.14.0
- matplotlib ≥3.9.0
- pandas ≥2.2.2
- anytree ≥2.13.0
- pyarrow ≥22.0.0

Dev dependencies: pytest, pytest-cov, ruff, marimo

## Package Structure

```
src/signaltour/
├── __init__.py          # Package entry point
├── Signal.py            # Signal module public API
├── Analysis.py          # Analysis module public API
├── Plot.py              # Plot module public API
├── _Signal_Module/      # Signal implementation
├── _Analysis_Module/    # Analysis implementation
├── _Plot_Module/        # Plot implementation
└── _Assist_Module/      # Shared utilities
```

## Testing

- Test fixtures defined in `test/conftest.py`
- `Sig` fixture provides a multi-component test signal (periodic + impulse + modulation)
- Tests use pytest framework

## Publishing

- Package published to PyPI as `signaltour-xcw`
- GitHub Actions workflow: `.github/workflows/python-publish.yml`
- Triggered on release publication or manual dispatch
- Uses trusted publishing (OIDC) for PyPI authentication
