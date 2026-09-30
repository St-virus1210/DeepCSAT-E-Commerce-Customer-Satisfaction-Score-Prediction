# Shopzilla Customer Satisfaction Prediction using Deep Learning ANN

## Project summary
This project predicts **CSAT Score (1–5)** for an e-commerce customer-support dataset using a feed-forward Deep Learning Artificial Neural Network.

Dataset: **85,907 records, 20 columns**

### Final test results
- Test Accuracy: **71.33%**
- Macro-F1: **0.252**
- Weighted-F1: **0.632**
- Majority-class baseline accuracy: **69.40%**
- Truncated SVD retained variance: **95.86%**

Because CSAT=5 represents about 69.4% of the records, accuracy alone is not sufficient. The report therefore emphasizes Macro-F1 and class-level metrics.

## Folder structure
```text
Shopzilla_CSAT_ANN_Project/
├── data/
├── models/
├── src/
│   ├── features.py
│   ├── model.py
│   └── train.py
├── app/
│   └── app.py
├── reports/
├── figures/
├── notebooks/
├── requirements.txt
├── README.md
└── run_app.bat
```

## How to run locally

### 1. Create an environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install packages
```bash
pip install -r requirements.txt
```

### 3. Use the supplied trained model
The `models/` directory already contains the preprocessing objects and ANN weights.

### 4. Launch the local application
```bash
streamlit run app/app.py
```

Open the local address displayed by Streamlit in your browser.

## Retraining
```bash
python src/train.py --data data/eCommerce_Customer_support_data.csv --epochs 8
```

## Methodology
1. Load and inspect data.
2. Parse support/order timestamps.
3. Engineer response time, order-to-issue time, calendar variables and text statistics.
4. Preserve missingness through explicit indicators and imputation.
5. One-hot encode categorical variables.
6. Reduce the sparse representation to 128 dense components using Truncated SVD.
7. Train a dense ANN with BatchNorm and Dropout.
8. Use a mild inverse-frequency class weighting strategy.
9. Evaluate with accuracy, Macro-F1, Weighted-F1 and a confusion matrix.

## Leakage policy
`Survey_response_Date` is not used as a model input because it occurs after the satisfaction response. Unique IDs and order IDs are also excluded from predictive features. Raw customer remarks are not fed directly into the model; only simple text statistics and lexicon counts are derived.

`issue_responded` and `connected_handling_time` are retained because this project is framed as **post-resolution CSAT prediction**, where service-response information is available.

## Important interpretation
The current dataset is highly imbalanced. The model's overall accuracy is above the majority-class baseline, but minority classes remain difficult to predict. This is a key finding, not something to hide: additional data for CSAT 2–4, richer text modeling, class-balanced sampling, or an ordinal-learning approach could improve the system.

## Video presentation demo
1. Explain the business problem.
2. Show the dataset and CSAT imbalance.
3. Explain feature engineering.
4. Explain the ANN architecture.
5. Show evaluation metrics and confusion matrix.
6. Run `streamlit run app/app.py`.
7. Enter a sample interaction.
8. Click **Predict CSAT** and explain the probability distribution.
9. Discuss limitations and future improvements.
