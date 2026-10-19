# 🚀 HOW TO RUN YOUR FRAUD DETECTION PROJECT

Complete step-by-step guide to run all 9 files in order.

---

## 📋 Prerequisites

Make sure you have:
- ✅ Python 3.8+ installed
- ✅ All 9 Python files created
- ✅ `requirements.txt` file

---

## 🔧 STEP 0: Set Up Environment & Install Dependencies

Run this FIRST to create a virtual environment and install all required libraries:

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Wait for installation to complete (1-2 minutes).

---

## ▶️ RUN THE PROJECT (9 Steps)

Run these commands in your terminal, **ONE BY ONE, in this order:**

### **STEP 1: Create Dataset**
```bash
python 1_data_collection.py
```

**What happens:**
- Creates sample dataset with 5,000 transactions
- Generates 95% genuine, 5% fraudulent transactions
- Saves `data.csv`

**Output:**
```
✅ Dataset saved as 'data.csv'
📁 File size: 5000 rows × 6 columns
```

---

### **STEP 2: Preprocess Data**
```bash
python 2_preprocessing.py
```

**What happens:**
- Loads `data.csv`
- Removes duplicates
- Checks for missing values
- Verifies data quality
- Saves `data_cleaned.csv`

**Output:**
```
✅ Cleaned dataset saved as 'data_cleaned.csv'
```

---

### **STEP 3: Exploratory Data Analysis (EDA)**
```bash
python 3_eda.py
```

**What happens:**
- Analyzes data patterns
- Creates visualizations
- Shows fraud vs genuine patterns
- Generates `eda_visualization.png`

**Output:**
```
✅ Visualizations saved as 'eda_visualization.png'
```

---

### **STEP 4: Handle Imbalanced Data (SMOTE)**
```bash
python 4_handle_imbalance.py
```

**What happens:**
- Splits data into train (80%) and test (20%)
- Applies SMOTE to balance training data
- Creates synthetic fraud samples
- Saves train-test split files

**Output:**
```
✅ Files saved:
   • X_train_balanced.csv
   • y_train_balanced.csv
   • X_test.csv
   • y_test.csv
```

**⭐ MOST IMPORTANT STEP!**

---

### **STEP 5: Feature Scaling**
```bash
python 5_feature_scaling.py
```

**What happens:**
- Normalizes all features to same scale
- Uses StandardScaler
- Saves scaled data
- Saves scaler for later use

**Output:**
```
✅ Files saved:
   • X_train_scaled.csv
   • X_test_scaled.csv
   • scaler.pkl
```

---

### **STEP 6: Train Model**
```bash
python 6_train_model.py
```

**What happens:**
- Trains Random Forest Classifier
- Uses 100 trees
- Saves trained model
- Shows feature importance

**Output:**
```
✅ Model saved as 'fraud_detection_model.pkl'
📊 Training accuracy: ~95%
📊 Test accuracy: ~93%
```

---

### **STEP 6b: Train XGBoost Model (Alternative)**
```bash
python 6b_train_xgboost_model.py
```

**What happens:**
- Trains XGBoost Classifier (100 boosting rounds)
- Saves trained model and feature importance chart
- Generates `xgboost_model.pkl` and `xgboost_feature_importance.png`

**Output:**
```
✅ XGBoost model saved as 'xgboost_model.pkl'
✅ Feature importance saved as 'xgboost_feature_importance.png'
📊 Training accuracy: 98.08%
📊 Test accuracy: 95.10%
📊 ROC-AUC: 0.9858
```

---

### **STEP 7: Evaluate Model**
```bash
python 7_evaluate_model.py
```

**What happens:**
- Makes predictions on test data
- Calculates Accuracy, Precision, Recall, F1, ROC-AUC
- Creates confusion matrix
- Generates evaluation visualizations
- Saves `model_evaluation.png`

**Output:**
```
📊 MODEL PERFORMANCE METRICS:
   Accuracy:  95.2%
   Precision: 92.8%
   Recall:    89.5%
   F1 Score:  0.910
   ROC-AUC:   0.945
```

---

### **STEP 8: Make Predictions**
```bash
python 8_make_predictions.py
```

**What happens:**
- Uses trained model to make predictions
- Tests on 3 sample transactions
- Shows fraud detection in action
- Demonstrates how to use in production

**Output:**
```
🔮 PREDICTION RESULTS:
   Example 1: ✓ GENUINE
   Example 2: 🚨 FRAUD
   Example 3: ⚠️ MEDIUM RISK
```

---

### **STEP 9: Launch Web App**
```bash
streamlit run 9_streamlit_app.py
```

**What happens:**
- Launches interactive web interface
- Opens at `http://localhost:8501`
- Can make real-time predictions
- Beautiful dashboard

**Output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

Then **open your browser** to `http://localhost:8501` and test the app!

---

## 📊 Expected Files After Running All Steps

```
project_folder/
├── 1_data_collection.py
├── 2_preprocessing.py
├── 3_eda.py
├── 4_handle_imbalance.py
├── 5_feature_scaling.py
├── 6_train_model.py
├── 6b_train_xgboost_model.py
├── 7_evaluate_model.py
├── 8_make_predictions.py
├── 9_streamlit_app.py
├── requirements.txt
│
└── Generated Files:
    ├── data.csv                       (from Step 1)
    ├── data_cleaned.csv               (from Step 2)
    ├── eda_visualization.png          (from Step 3)
    ├── X_train_balanced.csv           (from Step 4)
    ├── y_train_balanced.csv           (from Step 4)
    ├── X_test.csv                     (from Step 4)
    ├── y_test.csv                     (from Step 4)
    ├── X_train_scaled.csv             (from Step 5)
    ├── X_test_scaled.csv              (from Step 5)
    ├── scaler.pkl                     (from Step 5)
    ├── fraud_detection_model.pkl      (from Step 6)
    ├── xgboost_model.pkl              (from Step 6b)
    ├── xgboost_feature_importance.png (from Step 6b)
    └── model_evaluation.png           (from Step 7)
```

---

## ⏱️ How Long Does It Take?

- Step 1-2: 10 seconds
- Step 3: 20 seconds
- Step 4: 15 seconds
- Step 5: 5 seconds
- Step 6: 30 seconds
- Step 7: 10 seconds
- Step 8: 5 seconds
- **Total: ~1-2 minutes** for all steps

---

## 🆘 Troubleshooting

### "Command not found: python"
Use `python3` instead:
```bash
python3 1_data_collection.py
```

### "ModuleNotFoundError: No module named 'pandas'"
Install dependencies:
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: data.csv"
You skipped a step! Run in order:
```bash
python 1_data_collection.py  # Creates data.csv first!
python 2_preprocessing.py     # Then uses it
```

### "Port 8501 already in use"
Use a different port:
```bash
streamlit run 9_streamlit_app.py --server.port 8502
```

Then open: `http://localhost:8502`

### Streamlit app shows "Model not found"
Make sure you ran Step 6 first:
```bash
python 6_train_model.py
```

---

## 🎮 Testing the Web App

After Step 9, try these test transactions:

### Test 1: Genuine Transaction ✓
- Amount: ₹2000
- Time: 10:00 (10 AM)
- Merchant: 2
- Card: Credit
- International: No
- **Expected: ✓ GENUINE**

### Test 2: Suspicious Transaction ⚠️
- Amount: ₹8000
- Time: 03:00 (3 AM)
- Merchant: 1
- Card: Debit
- International: Yes
- **Expected: 🚨 FRAUD**

### Test 3: Medium Risk ⚠️
- Amount: ₹5000
- Time: 22:00 (10 PM)
- Merchant: 3
- Card: Credit
- International: Yes
- **Expected: ⚠️ MEDIUM RISK**

---

## 📝 What Each File Does

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| 1 | Create dataset | Nothing | data.csv |
| 2 | Clean data | data.csv | data_cleaned.csv |
| 3 | Analyze data | data_cleaned.csv | eda_visualization.png |
| 4 | Balance data with SMOTE | data_cleaned.csv | train/test CSVs |
| 5 | Scale features | train/test CSVs | X_train_scaled.csv, scaler.pkl |
| 6 | Train model | Scaled data | fraud_detection_model.pkl |
| 7 | Evaluate model | Model + test data | model_evaluation.png + metrics |
| 8 | Make predictions | Model + scaler | Predictions on samples |
| 9 | Web app | Model + scaler | Interactive interface |

---

## ✅ Quick Checklist

- [ ] Installed Python 3.8+
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] All 9 Python files created
- [ ] Ran Step 1: `python 1_data_collection.py`
- [ ] Ran Step 2: `python 2_preprocessing.py`
- [ ] Ran Step 3: `python 3_eda.py`
- [ ] Ran Step 4: `python 4_handle_imbalance.py`
- [ ] Ran Step 5: `python 5_feature_scaling.py`
- [ ] Ran Step 6: `python 6_train_model.py`
- [ ] Ran Step 7: `python 7_evaluate_model.py`
- [ ] Ran Step 8: `python 8_make_predictions.py`
- [ ] Ran Step 9: `streamlit run 9_streamlit_app.py`
- [ ] Opened web app at http://localhost:8501
- [ ] Tested predictions with sample transactions

---

## 🎉 You're Done!

You now have a complete, working fraud detection system! 

**Next Steps:**
1. Experiment with the web app
2. Try different transaction amounts
3. Modify thresholds in the code
4. Use real data from Kaggle
5. Deploy to cloud (Heroku, AWS, etc.)

---

## 📞 Need Help?

1. Check output messages for errors
2. Re-run the previous step
3. Make sure all files are in same directory
4. Verify `requirements.txt` is installed
5. Check Python version: `python --version`

---

**Happy Learning! 🚀**