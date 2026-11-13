# AgriPredict: Dynamic Crop Yield Prediction & Recommendation System

AgriPredict is an end-to-end, dynamic agricultural machine learning platform built with Python, Streamlit, Pandas, Scikit-learn, Plotly, Joblib, and Openpyxl.

The application allows users to upload custom agricultural datasets (CSV or Excel), automatically detects column semantics, handles data cleaning and preprocessing without data leakage, visualizes patterns with interactive Plotly graphics, benchmarks multiple machine learning regression models on held-out test data, and generates predictions and crop recommendation rankings.

---

## 🌟 Key Features

1. **Fully Dynamic Architecture**:
   - **Zero Hardcoding**: Works dynamically on any agricultural dataset with any crop varieties, regions, weather, or soil parameters.
   - **File Support**: Supports CSV (`.csv`) and Excel (`.xlsx`, `.xls`) with automatic encoding and delimiter detection.
   - **Semantic Role Detection**: Dynamically detects column roles (`Crop`, `Yield`, `Year`, `Region`, `Season`, `Cultivated Area`, `Production`, `Rainfall`, `Temperature`, `Soil`, `Fertilizer`, `Irrigation`) with user override controls.
   - **Dynamic Yield Calculation**: If Yield is absent but Production and Cultivated Area exist, calculates $\text{Yield} = \frac{\text{Production}}{\text{Cultivated Area}}$ while preventing division by zero.

2. **Rigorous Data Cleaning & Preprocessing**:
   - Preserves original raw dataset untouched.
   - Strips dirty strings, whitespace, and invalid number formatting.
   - Detects and removes duplicate rows.
   - Interquartile Range (IQR) outlier detection with inspect, cap, or remove options.
   - Prevents data leakage by encapsulating encoders and imputers inside isolated Scikit-learn Pipelines fit strictly on training splits.

3. **Exploratory Data Analysis (EDA)**:
   - Interactive Plotly visualizations for crop distributions, crop-wise yield variance, historical yield trends, regional comparisons, and correlation heatmaps.
   - Rainfall and temperature yield scatter plots with OLS regression trendlines.
   - Missing value distribution analysis.

4. **Automated Machine Learning (AutoML)**:
   - Evaluates multiple regression algorithms:
     - **Random Forest Regressor**
     - **Gradient Boosting Regressor**
     - **Decision Tree Regressor**
     - **Ridge Linear Regression**
   - Chronological or random train/test splitting.
   - Reports **MAE**, **RMSE**, and **R²** on held-out test data and training data to monitor overfitting.
   - Automatically selects and persists the best model bundle using `joblib`.
   - Dynamic feature importance ranking and plain-language performance explanations.

5. **Crop Yield Prediction**:
   - Form fields dynamically generated based on the exact features used by the trained model.
   - Out-of-distribution warnings when user inputs fall outside the historical training range.
   - Prediction session logging and downloadable CSV report.

6. **Best Crop Recommendation & Ranking**:
   - Simulates and compares candidate crops under uniform regional, soil, and weather conditions.
   - Ranks crops from highest to lowest predicted yield with interactive bar charts.
   - Agronomic and economic disclaimers emphasizing that highest physical yield does not always equate to highest farm profitability.

7. **Model Evaluation & Export Dashboard**:
   - Actual vs. Predicted scatter plot with $y = x$ reference line.
   - Residual error plots.
   - Export options for cleaned datasets, model comparison tables, prediction logs, ranking results, and trained `.joblib` pipelines.

---

## 📁 Project Structure

```text
crop-yield-prediction/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Comprehensive documentation
├── .gitignore                  # Git ignore file
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py       # File loading, validation & summary stats
│   ├── schema_detector.py      # Column role detection heuristics
│   ├── data_cleaner.py         # Leakage-free cleaning & yield derivation
│   ├── eda.py                  # Dynamic Plotly visualizations
│   ├── model_trainer.py        # Automated ML training & joblib persistence
│   ├── model_evaluator.py      # Diagnostics, residuals & evaluation
│   ├── predictor.py            # Form validation & inference execution
│   └── crop_recommender.py     # Crop ranking under uniform conditions
├── data/
│   ├── generate_sample_data.py # Sample data generation script
│   ├── agricultural_yield_dataset.csv
│   └── crop_production_derive_yield.xlsx
├── models/
│   └── best_crop_model.joblib  # Saved pipeline bundle
└── tests/
    ├── __init__.py
    ├── test_upload.py          # Loader and schema detector tests
    ├── test_preprocessing.py   # Cleaner, outlier & derivation tests
    └── test_prediction.py      # ML training, inference & ranking tests
```

---

## 🛠️ Technology Stack

- **Python**: 3.10+ (tested on Python 3.14)
- **Web UI**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
- **Visualizations**: [Plotly](https://plotly.com/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Excel Support**: [Openpyxl](https://openpyxl.readthedocs.io/)
- **Testing**: [Pytest](https://pytest.org/)

---

## ⚡ Installation & Setup Instructions

### 1. Open Terminal in VS Code
Open a Windows PowerShell terminal in the project directory:
```powershell
cd "d:\KL\SEM - 5\Data Science Project"
```

### 2. (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
To verify all modules and pipelines:
```powershell
python -m pytest tests/ -v
```

### 5. Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 📖 Step-by-Step User Guide

1. **Upload Dataset**:
   - Navigate to **📤 Upload Dataset & Schema**.
   - Click **"Browse files"** to upload your CSV/Excel file, or click **"Load Built-in Sample CSV"** for an instant demo.
   - Review the first 15 rows, column missing percentages, and verify detected column roles.
2. **Clean Data**:
   - Go to **🧹 Data Cleaning & Preprocessing**.
   - Select outlier options and duplicate removal.
   - If Yield is absent, check **"Calculate Yield = Production / Cultivated Area"**.
   - Click **"Execute Data Cleaning Pipeline"** and optionally download the cleaned dataset.
3. **Explore Data**:
   - Open **📊 Exploratory Data Analysis (EDA)** to inspect interactive yield distributions, time-series trends, and weather correlations.
4. **Train Models**:
   - Go to **🤖 Model Training & Comparison**.
   - Select your target variable (e.g. `Yield_Ton_Ha`) and input features.
   - Select Chronological or Random Split, then click **"Train & Compare Candidate Models"**.
5. **Predict Yield**:
   - On the **🌾 Predict Crop Yield** page, enter your farm's conditions and click **"Predict Crop Yield"**.
6. **Recommend Best Crop**:
   - Go to **🌟 Recommend Best Crop**, set local regional and weather conditions, and click **"Compare & Rank Candidate Crops"** to see which crop maximizes expected output.
7. **Evaluate Model**:
   - Go to **📈 Model Evaluation Dashboard** to inspect the Actual vs. Predicted plot, residual distributions, and agricultural domain warnings.
