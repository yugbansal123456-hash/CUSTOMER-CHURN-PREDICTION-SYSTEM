from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# LOAD MODEL FILES
model = pickle.load(open("final_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)

    original = data.copy()

    # =========================
    # PREPROCESSING
    # =========================
    if "customerID" in data.columns:
        data.drop("customerID", axis=1, inplace=True)

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    # Feature Engineering
    data["AvgCharges"] = data["TotalCharges"] / (data["tenure"] + 1)
    data["HighValueCustomer"] = (
        data["MonthlyCharges"] > data["MonthlyCharges"].median()
    ).astype(int)

    data = pd.get_dummies(data)
    data = data.reindex(columns=columns, fill_value=0)
    data = data.fillna(0)

    # =========================
    # PREDICTION
    # =========================
    scaled = scaler.transform(data)

    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)[:, 1]

    original["Prediction"] = pred
    original["Probability"] = prob

    # =========================
    # RISK SEGMENTATION
    # =========================
    def risk(p):
        if p < 0.3:
            return "Low"
        elif p < 0.7:
            return "Medium"
        else:
            return "High"

    original["Risk"] = original["Probability"].apply(risk)

    # =========================
    # SUMMARY
    # =========================
    total = len(original)
    high = (original["Risk"] == "High").sum()
    medium = (original["Risk"] == "Medium").sum()
    low = (original["Risk"] == "Low").sum()

    avg_prob = np.mean(prob)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    try:
        importances = model.named_estimators_['xgb'].feature_importances_
        feat_imp = pd.Series(importances, index=columns).sort_values(ascending=False).head(10)

        plt.figure()
        feat_imp.plot(kind='barh')
        plt.title("Top Features")
        plt.savefig("static/feature.png")
        plt.close()
    except:
        feat_imp = None

    # =========================
    # GRAPH 1: RISK BAR
    # =========================
    plt.figure()
    original["Risk"].value_counts().plot(kind='bar')
    plt.title("Risk Distribution")
    plt.savefig("static/risk_bar.png")
    plt.close()

    # =========================
    # GRAPH 2: PIE
    # =========================
    plt.figure()
    original["Risk"].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.ylabel("")
    plt.title("Risk Share")
    plt.savefig("static/risk_pie.png")
    plt.close()

    # =========================
    # TOP CUSTOMERS
    # =========================
    top = original.sort_values(by="Probability", ascending=False).head(5)

    # =========================
    # COST ANALYSIS
    # =========================
    cost = (high * 5000) + (medium * 2000)

    # SAVE FILE
    original.to_csv("result.csv", index=False)

    return render_template(
        "result.html",
        total=total,
        high=high,
        medium=medium,
        low=low,
        avg_prob=round(avg_prob, 3),
        cost=cost,
        preview=original.head().to_html(classes='data'),
        top=top.to_html(classes='data')
    )

@app.route('/download')
def download():
    return send_file("result.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)