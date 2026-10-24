import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

FEATURES = ["Study Hours", "Attendance", "Previous Score",
            "Assignments Score", "Gender_Male", "Parent Education"]


@st.cache_resource
def train_model():
    X_train = pd.read_csv("X_train.csv")[FEATURES]
    X_test = pd.read_csv("X_test.csv")[FEATURES]
    y_train = pd.read_csv("y_train.csv")["Final Marks"]
    y_test = pd.read_csv("y_test.csv")["Final Marks"]

    model = LinearRegression().fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, 100)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return model, X_train.mean(), mae, r2


model, avg_student, mae, r2 = train_model()

st.title(" Student Performance Predictor")
st.write("Enter a student's details to predict their final marks and see which factors matter.")

# ---------------- Inputs ----------------
st.sidebar.header("Student details")
study = st.sidebar.slider("Study hours per day", 1.0, 10.0, 5.0, 0.5)
attendance = st.sidebar.slider("Attendance (%)", 40, 100, 80)
previous = st.sidebar.slider("Previous score", 25, 100, 65)
assignments = st.sidebar.slider("Assignments score", 20, 100, 65)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
parent_edu = st.sidebar.selectbox("Parent education", ["High School", "Bachelor", "Master"])

edu_map = {"High School": 0, "Bachelor": 1, "Master": 2}
student = pd.DataFrame([[study, attendance, previous, assignments,
                         1 if gender == "Male" else 0, edu_map[parent_edu]]],
                       columns=FEATURES)

# ---------------- Prediction ----------------
prediction = float(np.clip(model.predict(student)[0], 0, 100))

tab1, tab2 = st.tabs(["Prediction", "About the model"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted final marks", f"{prediction:.1f}")
    c2.metric("Likely range", f"{max(prediction - mae, 0):.0f} to {min(prediction + mae, 100):.0f}")
    c3.metric("Class average (training)", f"{model.predict(avg_student.to_frame().T)[0]:.1f}")

    st.progress(int(prediction))

    if prediction < 50:
        st.error("⚠️ This student may need extra support. Consider early intervention.")
    elif prediction < 65:
        st.warning("🟡 This student is at moderate risk. Monitor progress and encourage study.")
    else:
        st.success("✅ This student is on track.")

    st.subheader("Why this prediction?")
    st.caption("Marks added or removed by each factor, compared with an average student.")
    contrib = pd.Series(model.coef_ * (student.iloc[0].values - avg_student.values),
                        index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(contrib.index, contrib.values,
            color=["#d9534f" if v < 0 else "#2e8b57" for v in contrib.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Marks added (+) or removed (-)")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown(f"""
**Model:** Linear Regression, trained on 800 students and tested on 200 unseen students.

| Metric | Test result |
|---|---|
| MAE (average error) | {mae:.2f} marks |
| R² score | {r2:.3f} |

**How it works**
1. The model learns how many marks each factor adds, from past students.
2. For a new student it adds up the effect of every factor.
3. The chart shows how each factor pushes the prediction up or down.

**Note:** the dataset is synthetic, built for demonstration. The support message thresholds
(below 50 and below 65) are example values, not official rules. Use predictions as a guide, not a final judgment.
""")