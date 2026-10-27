# Used Car Price Predictor

A beginner-friendly college mini-project comparing Linear Regression and
Polynomial Regression for estimating used-car resale prices. It uses Python,
Pandas, scikit-learn, Matplotlib, Seaborn, and Streamlit.

> **Important:** If `data/used_cars.csv` is not present, the training script
> generates 360 deterministic synthetic rows. Synthetic values are for
> demonstrating the code only. They are not real listings or reliable market
> prices, and the evaluation scores must not be presented as real-world accuracy.

## Project files

- `app.py` - Streamlit inputs, predictions, metrics, and charts.
- `train_model.py` - data creation/loading, cleaning, preprocessing, training,
  evaluation, and saving artifacts.
- `data/used_cars.csv` - generated on the first training run when no dataset is
  supplied; see `data/README.md` for the expected columns.
- `models/car_price_models.joblib` - generated trained pipelines and test data.
- `models/evaluation_metrics.json` - generated evaluation results.
- `requirements.txt` - Python packages needed by the project.

## Run on Windows in VS Code

1. Open this project folder in VS Code and open **Terminal > New Terminal**.
2. (Recommended) Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt instead and run
   `.venv\Scripts\activate.bat`, or continue with the environment's Python
   executable directly.

3. Install dependencies after reviewing `requirements.txt`:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the sample CSV and train both models:

   ```powershell
   python train_model.py
   ```

5. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

   Streamlit prints a local URL (usually `http://localhost:8501`) and normally
   opens it in your browser. Keep the terminal running while using the app.

No package installation or command execution is performed automatically by the
project files. If trained artifacts are missing, the dashboard displays the
training command needed before a prediction can be made.

## Using a real dataset

Replace `data/used_cars.csv` with a dataset you are allowed to use, then run
`python train_model.py` again. The CSV must contain these exact columns:

`company`, `model`, `year`, `km_driven`, `fuel`, `transmission`, `owner`,
`engine_cc`, `max_power_bhp`, `seats`, and `selling_price_inr`.

Prices must be in INR. The script validates required columns, removes duplicate
rows, converts the target to numeric, and drops rows with a missing target.
Feature missing values are imputed inside each model's training pipeline. The
input selectors use the dataset's company/model pairs; the specifications can
be entered in the sidebar. After replacing the CSV, retrain so the model
artifacts match it.

## How the project works

1. The script loads the local CSV. If absent, it writes a seeded synthetic
   demonstration dataset.
2. Duplicate rows are removed; invalid or missing target prices are discarded.
3. The data is divided into training and test sets. Imputation, numeric scaling,
   category encoding, and (for the polynomial model) feature expansion are
   fitted only as part of the training pipelines. This avoids test-data leakage.
4. Linear and polynomial pipelines are trained and evaluated on the same held-
   out test set. The fitted pipelines, predictions, and metrics are saved under
   `models/`.
5. The app loads those artifacts, accepts a car's details, and asks the selected
   pipeline for a price estimate. It also displays test metrics and charts.

## Regression in simple terms

**Linear Regression** estimates a straight-line relationship between the car's
features and price. With several features, this is a flat surface in a
multi-dimensional space. It is simple to explain but can miss curved patterns,
such as price changing nonlinearly with a car's age.

**Polynomial Regression** adds squared and interaction terms for numeric
features, then uses Linear Regression on the expanded features. This lets the
model represent some curves. Here the numeric features are expanded to degree
two; categorical data is one-hot encoded. More flexibility can also overfit,
especially with small datasets.

## Evaluation metrics

- **MAE (Mean Absolute Error):** average absolute difference between the
  estimate and actual price, in rupees. Lower is better.
- **MSE (Mean Squared Error):** average squared difference, in rupees squared.
  Large misses count more heavily. Lower is better.
- **RMSE (Root Mean Squared Error):** square root of MSE, so its unit is rupees.
  It is used here to choose the displayed best model; lower is better.
- **R² (R-squared):** how much of the variation in test prices the model
  explains relative to a constant baseline. Closer to 1 is generally better;
  it can be negative when the model performs worse than that baseline.

The app compares the metrics calculated from this run's held-out test set. It
does not assume polynomial regression always wins: whichever model has lower
test RMSE is reported as better for this particular split. Do not compare
synthetic-data results as evidence of real-market performance.