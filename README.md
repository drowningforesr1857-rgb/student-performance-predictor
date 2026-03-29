# 🎓 Student Performance Predictor

> **AI / Machine Learning Project** | Python | Scikit-learn | Pandas | Matplotlib

---

## 📌 Project Overview

This project uses **Machine Learning** to predict a student's final exam grade based on key academic and lifestyle factors. It compares two models — **Linear Regression** and **Random Forest** — to find the best predictor.

---

## 🎯 Problem Statement

Can we predict a student's final grade before the exam — using study habits, attendance, and previous performance?

---

## 📊 Dataset

- **300 student records** (synthetically generated based on real research patterns)
- **7 features:**

| Feature | Description |
|---|---|
| `study_hours` | Average study hours per day |
| `attendance_pct` | Attendance percentage |
| `previous_grade` | Grade from previous semester |
| `sleep_hours` | Average sleep hours per night |
| `internet_access` | Has internet at home (Yes/No) |
| `parent_education` | Highest education of parent |
| `final_grade` | **Target** — final exam score (0–100) |

---

## 🤖 Models Used

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression | ~5.2 | ~0.81 |
| **Random Forest** ✅ | **~3.8** | **~0.91** |

> Random Forest outperforms Linear Regression with higher accuracy.

---

## 📈 Visualizations

The script generates **4 charts** saved as `results_visualization.png`:
1. Study Hours vs Final Grade (scatter)
2. Actual vs Predicted Grades (RF model)
3. Feature Importance chart
4. Grade Distribution histogram

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 2. Run the project
python student_performance_predictor.py
```

---

## 🗂️ Project Structure

```
student_predictor/
│
├── student_performance_predictor.py   # Main ML script
├── student_data.csv                   # Auto-generated dataset
├── results_visualization.png          # Auto-generated charts
└── README.md                          # This file
```

---

## 💡 Key Findings

- **Study hours** is the most important predictor of final grade
- **Previous grade** and **attendance** are also strong indicators
- Students with internet access score on average **5 points higher**
- Random Forest achieves **91% accuracy** (R² = 0.91)

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-red)

---

## 👤 Author

**Ali Ahmed** — AI & Data Science Student  
📧 ali.ahmed@gmail.com  
🔗 linkedin.com/in/ali-ahmed

---

*Built as part of ICT Applications coursework — Lab 5: Digital Professional Identity*
