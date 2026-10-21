from flask import Flask, render_template, request
import pandas as pd
import os
import subprocess
import sys

app = Flask(__name__)

DATA_FILE = "object.csv"
ANALYSIS_FILE = "object_analysis.csv"


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(
        columns=[
            "id", "object", "category", "color",
            "location", "time", "status"
        ]
    )


@app.route("/")
def home():
    data = load_data()

    total_objects = len(data)
    lost_count = int((data["status"] == "Lost").sum())
    found_count = int((data["status"] == "Found").sum())

    most_common_object = (
        data["object"].mode().iloc[0]
        if not data.empty else "No data"
    )

    most_common_location = (
        data["location"].mode().iloc[0]
        if not data.empty else "No data"
    )

    cluster_count = 0

    if os.path.exists(ANALYSIS_FILE):
        analysis_data = pd.read_csv(ANALYSIS_FILE)

        if "pattern_cluster" in analysis_data.columns:
            cluster_count = analysis_data["pattern_cluster"].nunique()

    return render_template(
        "index.html",
        total_objects=total_objects,
        lost_count=lost_count,
        found_count=found_count,
        most_common_object=most_common_object,
        most_common_location=most_common_location,
        cluster_count=cluster_count
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    data = load_data()

    object_name = request.form.get("object", "").strip()
    category = request.form.get("category", "").strip()
    color = request.form.get("color", "").strip()
    location = request.form.get("location", "").strip()
    time_value = request.form.get("time", "").strip()
    status = request.form.get("status", "").strip()

    if not all([
        object_name, category, color,
        location, time_value, status
    ]):
        return "Please fill in all fields.", 400

    if status not in ["Lost", "Found"]:
        return "Invalid status.", 400

    # Find possible matching records BEFORE adding the new record.
    matches = pd.DataFrame()

    if status == "Lost" and not data.empty:
        matches = data[
            (data["status"].astype(str).str.strip().str.lower() == "found")
            & (data["object"].astype(str).str.strip().str.lower()
               == object_name.lower())
        ].copy()

        if not matches.empty:
            matches["match_score"] = (
                (matches["color"].astype(str).str.strip().str.lower()
                 == color.lower()).astype(int)
                +
                (matches["location"].astype(str).str.strip().str.lower()
                 == location.lower()).astype(int)
            )

            matches = matches[matches["match_score"] > 0]
            matches = matches.sort_values(
                by="match_score", ascending=False
            )

    # Add the new record to the CSV.
    new_id = (
        pd.to_numeric(data["id"], errors="coerce").max() + 1
        if not data.empty and "id" in data.columns
        and pd.to_numeric(data["id"], errors="coerce").notna().any()
        else 1
    )

    new_record = pd.DataFrame([{
        "id": int(new_id),
        "object": object_name,
        "category": category,
        "color": color,
        "location": location,
        "time": time_value,
        "status": status
    }])

    updated_data = pd.concat([data, new_record], ignore_index=True)
    updated_data.to_csv(DATA_FILE, index=False)

    # Run ML model
    result = subprocess.run(
        [sys.executable, "ml_model.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("ML model error:")
        print(result.stderr)

    # Read updated ML analysis
    analysis_data = pd.read_csv(ANALYSIS_FILE)

    new_id = new_record.iloc[0]["id"]

    cluster_rows = analysis_data[
        analysis_data["id"] == new_id
        ]

    cluster = None

    if not cluster_rows.empty:
        cluster = int(cluster_rows.iloc[0]["pattern_cluster"])

    return render_template(
        "result.html",
        record=new_record.iloc[0].to_dict(),
        matches=matches.to_dict(orient="records"),
        status=status,
        cluster=cluster
    )


if __name__ == "__main__":
    app.run(debug=True)