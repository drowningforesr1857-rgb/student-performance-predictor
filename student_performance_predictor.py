# ============================================================
#  Student Performance Predictor
#  AI / Machine Learning Project
#  Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── 1. Generate Realistic Dataset ───────────────────────────
np.random.seed(42)
n = 300

study_hours     = np.random.uniform(1, 10, n)
attendance_pct  = np.random.uniform(50, 100, n)
prev_grade      = np.random.uniform(40, 100, n)
sleep_hours     = np.random.uniform(4, 9, n)
internet_access = np.random.choice(["Yes", "No"], n, p=[0.75, 0.25])
parent_edu      = np.random.choice(["Primary", "Secondary", "University"], n)

# Final grade formula (realistic weights)
noise = np.random.normal(0, 4, n)
final_grade = (
    study_hours * 3.5 +
    attendance_pct * 0.25 +
    prev_grade * 0.30 +
    sleep_hours * 1.2 +
    np.where(internet_access == "Yes", 5, 0) +
    noise
)
final_grade = np.clip(final_grade, 0, 100)

df = pd.DataFrame({
    "study_hours":     study_hours,
    "attendance_pct":  attendance_pct,
    "previous_grade":  prev_grade,
    "sleep_hours":     sleep_hours,
    "internet_access": internet_access,
    "parent_education":parent_edu,
    "final_grade":     final_grade
})

df.to_csv("student_data.csv", index=False)
print("✅ Dataset created: 300 students, 7 features")
print(df.describe().round(2))

# ── 2. Preprocessing ─────────────────────────────────────────
le = LabelEncoder()
df["internet_access"]  = le.fit_transform(df["internet_access"])   # Yes=1, No=0
df["parent_education"] = le.fit_transform(df["parent_education"])  # encoded

X = df.drop("final_grade", axis=1)
y = df["final_grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n✅ Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 3. Train Models ──────────────────────────────────────────
lr  = LinearRegression()
rf  = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("\n📊 MODEL RESULTS")
print("─" * 40)
print(f"  Linear Regression  → MAE: {mean_absolute_error(y_test, lr_pred):.2f}  |  R²: {r2_score(y_test, lr_pred):.3f}")
print(f"  Random Forest      → MAE: {mean_absolute_error(y_test, rf_pred):.2f}  |  R²: {r2_score(y_test, rf_pred):.3f}")

# ── 4. Visualizations ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Student Performance Predictor — Analysis", fontsize=15, fontweight="bold", y=1.01)

# (a) Study Hours vs Final Grade
axes[0,0].scatter(df["study_hours"], y, alpha=0.5, color="#0a66c2", edgecolors="white", s=50)
axes[0,0].set_xlabel("Study Hours / Day")
axes[0,0].set_ylabel("Final Grade")
axes[0,0].set_title("Study Hours vs Final Grade")

# (b) Predicted vs Actual (RF)
axes[0,1].scatter(y_test, rf_pred, alpha=0.6, color="#00b050", edgecolors="white", s=50)
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0,1].set_xlabel("Actual Grade")
axes[0,1].set_ylabel("Predicted Grade")
axes[0,1].set_title("Actual vs Predicted (Random Forest)")

# (c) Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
colors = ["#0a66c2" if v == feat_imp.max() else "#BDD7EE" for v in feat_imp]
feat_imp.plot(kind="barh", ax=axes[1,0], color=colors)
axes[1,0].set_title("Feature Importance (Random Forest)")
axes[1,0].set_xlabel("Importance Score")

# (d) Grade distribution
axes[1,1].hist(y, bins=20, color="#0a66c2", edgecolor="white", alpha=0.85)
axes[1,1].axvline(y.mean(), color="red", linestyle="--", label=f"Mean: {y.mean():.1f}")
axes[1,1].set_xlabel("Final Grade")
axes[1,1].set_ylabel("Number of Students")
axes[1,1].set_title("Grade Distribution")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("results_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Visualization saved: results_visualization.png")

# ── 5. Predict a New Student ─────────────────────────────────
print("\n🎯 SAMPLE PREDICTION")
print("─" * 40)
new_student = pd.DataFrame([{
    "study_hours":     6.5,
    "attendance_pct":  85.0,
    "previous_grade":  72.0,
    "sleep_hours":     7.0,
    "internet_access": 1,
    "parent_education":2
}])
predicted = rf.predict(new_student)[0]
print(f"  Study: 6.5h/day | Attendance: 85% | Prev Grade: 72")
print(f"  ➜  Predicted Final Grade: {predicted:.1f} / 100")
print("\n✅ Project complete!")
