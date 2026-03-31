# Contributing to This Project

## Architecture Principles

This project follows strict architectural principles to maintain code quality, reproducibility, and maintainability:

1. **Separation of Concerns**: Each module has exactly ONE responsibility
   - `data_preprocessing.py` ← data loading and cleaning
   - `feature_engineering.py` ← feature transformation
   - `train.py` ← model training
   - `evaluate.py` ← model evaluation
   - `predict.py` ← prediction on new data
   - `persistence.py` ← save/load artifacts

2. **No Circular Dependencies**: This forms a clean dependency hierarchy:
   ```
   config ← utils ← preprocessing ← features ← train ← evaluate ← predict
   ```

3. **No Data Leakage**: Training and prediction code are completely isolated
   - Training: `fit_transform()` on training data
   - Prediction: `transform()` only (never fits)

4. **Encapsulation**: Each function
   - Accepts all inputs as parameters (no global state)
   - Returns results explicitly
   - Validates inputs and raises clear errors
   - Includes comprehensive logging

## Code Structure Requirements

### Function Design
```python
def my_function(param1: Type1, param2: Type2) -> ReturnType:
    """
    Clear docstring with purpose, args, returns, and raises.
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description of return
        
    Raises:
        ValueError: When input is invalid
    """
    # 1. Validate inputs
    if param1 is None:
        raise ValueError("param1 cannot be None")
    
    # 2. Log start
    logger.info(f"Starting operation with param1={param1}")
    
    try:
        # 3. Main logic
        result = do_something(param1, param2)
        
        # 4. Log completion
        logger.info("Operation completed successfully")
        
        # 5. Return result
        return result
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
```

### Module Structure
```python
"""
Module docstring explaining this module's purpose.

Clear explanation of what it does and where it fits in the pipeline.
"""

import logging
from typing import Type1, Type2

# Create module logger
logger = logging.getLogger(__name__)

# ============================================================================
# FUNCTIONS (organized logically with section headers)
# ============================================================================

def function1(...):
    """Purpose and signature."""
    pass

def function2(...):
    """Purpose and signature."""
    pass

# If module has a main entry point
if __name__ == "__main__":
    main()
```

## Before Making Changes

### Ask These Questions

1. **Where does this code belong?**
   - Does it fit clearly in one existing module?
   - If multiple modules are affected, you may be violating separation of concerns

2. **Does this change data leakage risk?**
   - Training code should never refit during prediction
   - Prediction code should never save artifacts

3. **Can this be tested in isolation?**
   - If your function depends on global state or side effects, refactor it
   - Pure functions (no side effects beyond their return) are best

4. **Will this impact reproducibility?**
   - All randomness should be controlled by `RANDOM_STATE` from `config.py`
   - No hardcoded values allowed

### What NOT to Do

❌ **DO NOT** add logic to the wrong module:
- Don't add training logic to `evaluate.py`
- Don't add evaluation to `train.py`
- Don't add feature engineering to `data_preprocessing.py`

❌ **DO NOT** create circular imports:
- Don't import `train.py` in `predict.py`
- Don't import `evaluate.py` in `train.py`

❌ **DO NOT** use global variables:
```python
# Bad
RANDOM_STATE = 42  # In train.py
def train_model(X, y):
    model.fit(X, y, ...)  # Uses global RANDOM_STATE

# Good
from src.config import RANDOM_STATE
def train_model(X, y, random_state=RANDOM_STATE):
    model.fit(X, y, ...)  # Uses parameter
```

❌ **DO NOT** hardcode paths:
```python
# Bad
df = pd.read_csv('/Users/name/project/data/file.csv')

# Good
from src.config import DATA_PATH
df = pd.read_csv(DATA_PATH)
```

## Making a Change

### Step 1: Identify the Module
- If changing how data is cleaned → `data_preprocessing.py`
- If changing feature transformations → `feature_engineering.py`
- If changing model algorithm → `train.py`
- If adding new metrics → `evaluate.py`

### Step 2: Make the Change
```python
# Add comprehensive docs
# Add input validation
# Add logging
# Keep function focused and pure
```

### Step 3: Update Tests
Add or update tests in `test_pipeline.py`:
```python
def test_your_change():
    """Test that your change works correctly."""
    # Setup
    input_data = create_mock_data()
    
    # Execute
    result = your_function(input_data)
    
    # Verify
    assert result.shape[0] == expected_rows
    assert 'expected_column' in result.columns
```

### Step 4: Verify No Regressions
```bash
# Run all tests
pytest src/test_pipeline.py -v

# Run with coverage
pytest src/test_pipeline.py -v --cov=src

# Run the full pipeline
python -m src.main
```

### Step 5: Update Configuration if Needed
If your change introduces new parameters:
```python
# In config.py
MY_NEW_PARAMETER = 42
```

Then import it:
```python
# In your module
from src.config import MY_NEW_PARAMETER
```

### Step 6: Update Documentation
If your change affects usage, update these:
- Function docstrings
- `README.md` usage instructions
- New `.md` files if adding significant features

## Common Change Patterns

### Adding a New Preprocessing Step
1. Add function to `feature_engineering.py`
2. Add it to the pipeline in `build_preprocessing_pipeline()`
3. Add test in `test_pipeline.py`
4. Run full pipeline to verify

### Changing the Model Algorithm
1. Update `train.py` function
2. Update hyperparameters in `config.py`
3. Update tests in `test_pipeline.py`
4. Run full pipeline and compare metrics

### Adding New Evaluation Metric
1. Add metric computation to `evaluate.py`
2. Return it from `evaluate_model()`
3. Update tests
4. Check `main.py` logging if needed

### Fixing a Bug
1. Add a test that fails with the current code
2. Fix the bug
3. Verify test passes
4. Run full test suite

## Performance Considerations

- Use `n_jobs=-1` in model training to utilize all cores
- Log performance-critical operations: `logger.debug(...)`
- Profile bottlenecks before adding complexity

## Logging Guidelines

```python
# Use appropriate levels
logger.debug("Detailed info for debugging")           # Development
logger.info("Important milestone completed")          # Pipeline flow
logger.warning("Something unexpected but recoverable") # Anomalies
logger.error("Operation failed")                      # Failures
```

## Code Review Checklist

Before submitting:
- [ ] Function validates all inputs
- [ ] Function has comprehensive docstring
- [ ] No hardcoded values (all in config.py)
- [ ] Logging added for important operations
- [ ] Tests written and passing
- [ ] No circular imports
- [ ] Follows module's responsibility pattern
- [ ] Error messages are clear and actionable

## Questions?

Refer to:
- `README.md` for project overview
- Existing modules for pattern examples
- `test_pipeline.py` for testing patterns

---

**Remember**: Structure enables collaboration. When responsibilities are clear, changes are safe.
