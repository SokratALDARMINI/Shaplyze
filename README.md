## 📦 Installation (Python 3.11+ Recommended)

We recommend using a virtual environment with [Conda](https://docs.conda.io) or `venv`.

### ✅ 1. Create and activate environment
```bash
conda create -n shaplyze python=3.11
conda activate shaplyze
```

### ✅ 2. Install required packages
```bash
pip install "numpy<2.0" pandas scikit-learn
pip install folktables liac-arff
pip install cvxpy ecos mosek
pip install category_encoders
pip install matplotlib seaborn
pip install pycddlib-standalone
pip install dit
```

---

> 🔧 **Note on `pycddlib`**: If installation fails due to missing `setoper.h` or `gmp.h`, you can use the pure Python fallback:
> ```bash
> pip install pycddlib-standalone
> ```

> ⚠️ **Note on `dit`**: The `dit` library depends on `pycddlib` and `pypoman`. If you're using `pycddlib-standalone`, it should work transparently. In rare cases, you may need to install `dit` from source:
> ```bash
> git clone https://github.com/dit/dit.git
> cd dit
> pip install . --no-deps
> ```

> 📝 **Note on `mosek`**: To use the MOSEK solver, you need a valid license. You can obtain a free academic license from [https://www.mosek.com](https://www.mosek.com).
---
## 🔍 Examples Usage:

This example demonstrates how to use the `SurrogateMeasure` class to evaluate the mutual information between features and the sensitive attribute using predicted outputs from a classifier.

### Step 1: Prepare a Simple Dataset
```python
import pandas as pd
import numpy as np

# Generate some mock data
np.random.seed(0)
Xc = np.random.randn(100)
Xg = np.random.randn(100)
A = np.random.rand(100)
Y = (Xc + Xg + A + 0.1*np.random.randn(100) > 0).astype(int)

# Construct DataFrame
df = pd.DataFrame({
    'X1': Xc,
    'X2': Xg,
    'A': A,
    'Y': Y
})

# Quantize continuous variables into 2 equal-width bins
for col in ['X1', 'X2', 'A', 'Y']:
    df[col] = pd.cut(df[col], bins=2, labels=False)

# Define inputs for the measure
features = ['X1', 'X2']
sensitive_attribute = ['A']
output = ['Y']
bins = (2, 2, 2, 2)
```

### Step 2: Non-classifier Based Measures
```python
from src.measures import (
    I_Xs_A, I_Y_Xs_given_AXsc, SI_A_Xs_Y, SI_Xs_A_Y,
    I_Xs_A_given_Y, I_A_Xs_times_IAXs_given_Y_times_SI_Y_Xs_A,
    HSIC_Xs_Y, NOCCO_Xs_Y, MMD_Xs_Y, MMD_Xs_A
)
from src.shaplyze import ShaplyzeEstimator

# Example with mutual information I(Xs;A)
measure = I_Xs_A(df, features, sensitive_attribute, output, bins)
estimator = ShaplyzeEstimator(measure)
shap_values = estimator.get_sh_values()
print("Shapley values (I_Xs_A):", shap_values)
```

### Step 3: Classifier-based Measures
```python
from sklearn.neural_network import MLPClassifier
from src.classifier import AccuracyMeasure, DPMeasure, EOMeasure

clf = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=0)

# Accuracy Measure example
measure_acc = AccuracyMeasure(
    df, features, sensitive_attribute, output,
    classifier=clf, mode="drop", categorical_features=None
)
estimator_acc = ShaplyzeEstimator(measure_acc)
shap_values_acc = estimator_acc.get_sh_values()
print("Shapley values (Accuracy):", shap_values_acc)
```

### Step 4: Surrogate Measure Example
```python
from src.measures import SurrogateMeasure

# Create SurrogateMeasure using drop-out mode
surrogate_mi = SurrogateMeasure(
    df, features, sensitive_attribute, output,
    measure_class=I_Xs_A,
    mode="drop",
    X=features, A=sensitive_attribute, Y=output,
    bins=bins
)

# Evaluate using ShaplyzeEstimator
estimator = ShaplyzeEstimator(surrogate_mi)
shap_values = estimator.get_sh_values()
print("Shapley values from surrogate measure:", shap_values)
```


## 📁 Datasets (for benchmarking)

To run the examples in the toolkit, you must manually download the following datasets and place them inside the `datasets/` folder at the root of the project.

### 🔸 1. COMPAS Dataset
- **Required file**: `datasets/compas-scores-two-years.csv`
- **Source**: [ProPublica GitHub Repository](https://github.com/propublica/compas-analysis)
- **Direct download**: [compas-scores-two-years.csv](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)

---

### 🔸 2. Adult Dataset
- **Required file**: `datasets/adult.csv`
- **Source**: UCI Machine Learning Repository  
- **URL**: [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)

---

### 🔸 3. Census Income KDD Dataset (ARFF)
- **Required file**: `datasets/dataset.arff`
- **Source**: UCI KDD Census Income Dataset  
- **URL**: [https://kdd.ics.uci.edu/databases/census-income/census-income.data](https://kdd.ics.uci.edu/databases/census-income/)

---

### 🔸 4. Heritage Health Dataset
- **Required files**:
  - `datasets/Claims.txt`
  - `datasets/Members.txt`
- **Source**: [Heritage Health Prize on Kaggle](https://www.kaggle.com/c/hhp)
- **Note**: You must be logged into Kaggle to download these files.

---

📌 Ensure all files are placed inside the `datasets/` directory using the exact filenames listed above.

## 📁 Run Benchmarking
To run the full benchmarking pipeline for any dataset, use:
```bash
python benchmarking.py --dataset <id>
```
where `<id>` is the index (0 to 6) corresponding to a specific dataset in the script.
