# Batch-1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
warnings.filterwarnings('ignore')
df = pd.read_csv('Bank Customer Churn Prediction.csv')
print(df.head())
print(df.info())
print(df.describe())
for col in df.select_dtypes(include='object').columns:
    print(f"\nUnique values in '{col}':\n", df[col].unique())
sns.countplot(data=df, x='churn')
plt.title('Churn Distribution')
plt.show()
categorical_cols = ['country', 'gender', 'credit_card', 'active_member']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, hue='churn')
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Churn')
    plt.tight_layout()
    plt.show()
numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, hue='churn', bins=30, kde=True)
    plt.title(f'{col} vs Churn')
    plt.tight_layout()
    plt.show()
print("Columns in DataFrame:", df.columns.tolist())
le = LabelEncoder()
if 'gender' in df.columns:
    df['gender'] = le.fit_transform(df['gender'])
if 'country' in df.columns:
    df = pd.get_dummies(df, columns=['country'], drop_first=True)
if 'customer_id' in df.columns:
    df.drop(columns=['customer_id'], inplace=True)
if 'churn' in df.columns:
    X = df.drop('churn', axis=1)
    y = df['churn']
else:
    raise KeyError("Target column 'churn' not found in DataFrame.")
feature_columns = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1]))
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1]))
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost")
importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Top 10 Important Features (Random Forest)')
plt.show()
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df, explorative=True)
    profile.to_file("customer_report.html")
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'ydata-profiling'])
try:
    import gradio as gr
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'gradio'])
    import gradio as gr
def predict_churn(credit_score, country, gender, age, tenure, balance, products_number, has_credit_card,
is_active_member, estimated_salary):
    gender = 1 if gender == "Male" else 0
    input_data = {
        'credit_score': credit_score,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': has_credit_card,
        'active_member': is_active_member,
        'estimated_salary': estimated_salary
    }
    for col in feature_columns:
        if col.startswith("country_"):
            input_data[col] = 0
    country_col = f"country_{country}"
    if country_col in feature_columns:
        input_data[country_col] = 1
    else:
        return f"❌ Country '{country}' not recognized. Try: {', '.join([col.split('_')[1] for col in feature_columns if
'country_' in col])}"
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prob = xgb_model.predict_proba(input_scaled)[0][1]
    return f"✅ Churn Probability: {prob:.2%}"
inputs = [
    gr.Number(label="Credit Score"),
    gr.Textbox(label="Country (e.g., France, Germany, Spain)"),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Number(label="Age"),
    gr.Number(label="Tenure"),
    gr.Number(label="Balance"),
    gr.Number(label="Products Number"),
    gr.Radio([0, 1], label="Has Credit Card"),
    gr.Radio([0, 1], label="Is Active Member"),
    gr.Number(label="Estimated Salary"),
]
output = gr.Textbox(label="Churn Prediction")
gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=output,
    title="🔮 Bank Customer Churn Predictor",
    description="Enter customer details to predict churn probability using XGBoost."
).launch(share=True)
