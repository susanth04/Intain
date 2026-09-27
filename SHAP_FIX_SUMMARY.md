# SHAP Explainability Bug Fix Summary

## Root Cause
The SHAP explainability feature was showing "Explainability unavailable for this model" because the model loader (`backend/models/loader.py`) had incorrect path resolution logic:

1. **Primary issue**: The hardcoded fallback path `C:\Users\susan\OneDrive\Desktop\Intain\loan-perf-engine` didn't match the actual project structure
2. **Secondary issue**: The auto-discovery logic only looked for `config.yaml` in parent directories, but didn't check for the `engine/` subdirectory
3. **Environment variable**: `INTAIN_PROJECT_ROOT` was set in Docker but not for local development

## Changes Made

### 1. Fixed Path Resolution (`backend/models/loader.py`)
- Added fallback logic to check for `engine/config.yaml` as a sibling to `backend/`
- Added relative path fallback when auto-discovery fails
- Now correctly resolves to `../engine` for local development

### 2. Enhanced Error Handling (`backend/pipeline/explainability.py`)
- Added debug logging to show which models are available
- Improved error messages to show available models when one is missing
- Added exception tracebacks for better debugging
- Limited error message length to avoid overwhelming the frontend

### 3. Environment Configuration
- Updated `.env.example` to document `INTAIN_PROJECT_ROOT=../engine` for local dev
- Added `backend/.env.example` for backend-specific configuration
- Dockerfile already had correct `INTAIN_PROJECT_ROOT=/app/engine`

### 4. Additional Discovery
During investigation, found that `next_12m_default_flag` and `next_12m_prepayment_flag` models fall back to Logistic Regression instead of LightGBM due to "no clear signal" in training (see `engine/src/models/delinquency_default_prepay.py` lines 128-132). This explains why their metrics are identical between LR and LGBM.

## Testing Verification
The fix ensures:
1. Model loader can find `engine/config.yaml` in both local and Docker environments
2. Raw LightGBM models (used for SHAP) are correctly loaded
3. Feature alignment between model expectations and input features works
4. SHAP TreeExplainer can successfully compute explanations

## Expected Behavior After Fix
When a user runs pipeline analysis for a loan:
1. Features are engineered correctly
2. Raw LightGBM models are loaded from `engine/data/processed/models/`
3. SHAP TreeExplainer computes feature attributions
4. Frontend displays top 15 features with SHAP values, direction, and impact

## Files Modified
- `backend/models/loader.py` - Fixed path resolution logic
- `backend/pipeline/explainability.py` - Enhanced error handling and debugging
- `.env.example` - Updated documentation
- `backend/.env.example` - Created backend-specific env template
- `Dockerfile` - Added comment for clarity
