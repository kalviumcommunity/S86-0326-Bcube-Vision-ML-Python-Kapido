# Virtual Environment Setup Guide

## Overview

Your ML project now has a **virtual environment** (`venv/`) that isolates Python dependencies for this project only. This ensures:

✅ **Reproducibility** - Same environment on any machine
✅ **Isolation** - No conflicts with other projects  
✅ **Portability** - Easy to share and deploy
✅ **Cleanliness** - Keeps your system Python untouched

## Quick Start

### Windows
```bash
activate_env.bat
```

Or manually:
```powershell
venv\Scripts\activate
```

### macOS/Linux
```bash
source activate_env.sh
```

Or manually:
```bash
source venv/bin/activate
```

## What Was Installed

Once the virtual environment is activated, you have these packages available:

### Core ML Stack
- **pandas** (3.0.2) - Data manipulation
- **numpy** (2.4.4) - Numerical computing
- **scikit-learn** (1.8.0) - Machine learning algorithms
- **scipy** (1.17.1) - Scientific computing
- **joblib** (1.5.3) - Model serialization

### Testing & Development
- **pytest** (9.0.2) - Unit testing framework
- **pytest-cov** (7.1.0) - Code coverage reporting

### Dependencies (automatically installed)
- **python-dateutil**, **tzdata** - Date/time handling
- **threadpoolctl** - Thread pool control
- **coverage** - Coverage measurement

## Working in the Virtual Environment

### Activate Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

Your prompt will change to show `(venv)`:
```
(venv) C:\project>  # Windows
(venv) user@machine:~/project$  # macOS/Linux
```

### View Installed Packages
```bash
pip list
```

Example output:
```
Package           Version
----------------- -----------
pandas            3.0.2
numpy             2.4.4
scikit-learn      1.8.0
pytest            9.0.2
...
```

### Install New Package
```bash
pip install package_name
```

Example:
```bash
pip install matplotlib
```

### Update Package
```bash
pip install --upgrade package_name
```

### Run Python Scripts
```bash
# Inside activated environment
python src/main.py
python -m src.main
pytest src/test_pipeline.py -v
```

### Deactivate Environment
```bash
deactivate
```

You return to your system's default Python.

## Important Files

| File/Directory | Purpose |
|---|---|
| `venv/` | **DO NOT COMMIT** - Machine-specific environment |
| `requirements.txt` | **COMMIT** - Lists all dependencies |
| `activate_env.bat` | Windows activation helper |
| `activate_env.sh` | macOS/Linux activation helper |

## Project Structure Now

```
S86-0326-Bcube-Vision-ML-Python-Kapido/
│
├── venv/                          ← Virtual environment (isolated)
│   ├── Scripts/                   ← Executables (python, pip, pytest)
│   ├── Lib/                       ← Installed packages
│   └── Include/                   ← Header files
│
├── src/                           ← Your code
├── data/                          ← Your data
├── models/                        ← Trained models
├── requirements.txt               ← Dependency list (for git)
├── activate_env.bat              ← Windows activation
├── activate_env.sh               ← macOS/Linux activation
└── README.md                      ← Documentation
```

## Reproducibility with requirements.txt

The `requirements.txt` file locks dependency versions:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
pytest>=7.0.0
pytest-cov>=3.0.0
```

### For Anyone Cloning This Project

```bash
# 1. Clone the repository
git clone <repo-url>
cd S86-0326-Bcube-Vision-ML-Python-Kapido

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
pip list
```

Now they have the exact same environment as you!

### Update requirements.txt with New Packages

If you add new packages in development:

```bash
# Install new package
pip install some_new_package

# Update requirements.txt with exact versions
pip freeze > requirements.txt

# Commit the updated file
git add requirements.txt
git commit -m "Add new_package dependency"
```

## Why Virtual Environments Matter

### Without Virtual Environment ❌
```
System Python
├── pandas 1.0  (for Project A)
├── pandas 2.0  (for Project B)
└── Conflict!   (Projects break)
```

### With Virtual Environment ✅
```
Project A (venv)
└── pandas 1.0 (isolated)

Project B (venv)
└── pandas 2.0 (isolated)

Both work perfectly!
```

## Common Tasks

### Run the ML Pipeline
```bash
# Activate environment first
venv\Scripts\activate  # Windows

# Run pipeline
python -m src.main

# View results in logs/pipeline.log
```

### Run Tests
```bash
# Activate environment
venv\Scripts\activate

# Run all tests
pytest src/test_pipeline.py -v

# Run with coverage
pytest src/test_pipeline.py -v --cov=src
```

### Make Predictions
```bash
# Activate environment
venv\Scripts\activate

# Generate predictions
python -m src.predict
```

### Add Development Tools
```bash
# Activate environment
venv\Scripts\activate

# Install additional tools
pip install jupyter                      # Notebooks
pip install black                        # Code formatting
pip install flake8                       # Code linting
pip install mypy                         # Type checking

# Update requirements.txt
pip freeze > requirements.txt
```

## Troubleshooting

### Command: `python -m venv venv` not working

**Issue**: venv module not found

**Solution**: 
```bash
# Ensure Python 3.9+ installed
python --version

# On Windows, you might need:
py -m venv venv

# On Ubuntu/Debian:
sudo apt-get install python3-venv
python3 -m venv venv
```

### Packages Not Found After Activation

**Issue**: After activating, pip or python commands not found

**Solution**:
- Ensure you ran the activation script
- Check prompt shows `(venv)`
- On Windows, try: `venv\Scripts\activate.bat`
- On macOS/Linux, try: `source venv/bin/activate`

### Permission Denied on macOS/Linux

**Issue**: cannot execute activate script

**Solution**:
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### Virtual Environment Taking Up Space

**Issue**: venv/ directory is large

**Solution**: This is normal (100MB-500MB depending on packages)
- Don't commit `venv/` to git (already in .gitignore)
- Can safely delete and recreate: `rm -rf venv && python -m venv venv && pip install -r requirements.txt`

### Wrong Python Being Used

**Issue**: `python --version` shows wrong version

**Solution**:
- Ensure venv is activated
- Check prompt includes `(venv)`
- Run: `which python` (macOS/Linux) or `where python` (Windows)
- Should show path inside `venv/` directory

## Environment Status Checklist

After setup, verify:

- ✅ Virtual environment created: `venv/` directory exists
- ✅ Environment activates: `(venv)` appears in prompt
- ✅ Packages installed: `pip list` shows all dependencies
- ✅ Tests pass: `pytest src/test_pipeline.py -v` runs successfully
- ✅ Pipeline runs: `python -m src.main` executes without errors
- ✅ .gitignore configured: `venv/` in .gitignore

## Next Steps

1. **Activate the environment**
   ```bash
   # Windows
   activate_env.bat
   # Or macOS/Linux
   source activate_env.sh
   ```

2. **Verify everything works**
   ```bash
   pip list
   pytest src/test_pipeline.py -v
   ```

3. **Run the ML pipeline**
   ```bash
   python -m src.main
   ```

4. **Share with teammates**
   - They clone the repo
   - They run the same steps (create venv, activate, pip install -r requirements.txt)
   - They get identical environment!

## Key Concepts

| Concept | Meaning |
|---------|---------|
| **Activation** | Switch terminal to use venv's Python |
| **Deactivation** | Switch back to system Python |
| **pip** | Package installer (only works in activated venv) |
| **requirements.txt** | Lock file for reproducibility |
| **Isolation** | venv packages don't affect system or other projects |

## Resources

- [Python venv Documentation](https://docs.python.org/3/library/venv.html)
- [pip Documentation](https://pip.pypa.io/)
- [requirements.txt Format](https://pip.pypa.io/en/latest/reference/requirements-file-format/)

---

**Your ML project is now completely isolated and reproducible!**
