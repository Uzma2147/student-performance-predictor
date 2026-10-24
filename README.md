#  Student Performance Predictor

A student performance predictor that cleans and explores student data, then compares Linear Regression, Decision Tree, and Random Forest models using MAE, MSE, and R². The best model (Linear Regression, R² of 0.88) powers a Streamlit app that predicts final marks and shows which factors help or hurt each student.

 **Source code:** https://github.com/YOUR-USERNAME/student-performance-predictor


##  Problem Statement

Educational institutions want to identify students who may need extra support **before** the exam, not after. This project builds a regression model that predicts a student's final marks from academic and personal factors, so teachers can spot at-risk students early.

##  Dataset

The dataset is "synthetic" (generated in code) with 1,000 students, so the focus is on the method and pipeline.

| Column | Description |

| Study Hours | Study time per day (1 to 10) |
| Attendance | Attendance percentage (40 to 100) |
| Previous Score | Previous exam score (25 to 100) |
| Assignments Score | Assignment score (20 to 100) |
| Gender | Male / Female |
| Parent Education | High School / Bachelor / Master |
| **Final Marks** | **Target variable (0 to 100)** |

Missing values were added on purpose in Attendance, Previous Score, and Parent Education to practice real preprocessing.

##  Project Workflow

1. Data collection: generated a synthetic dataset with a hidden formula plus random noise.
2. Preprocessing:
   - Train/test split first (80% / 20%) to avoid data leakage
   - Missing values filled using the training median (numbers) and mode (text)
   - Categorical columns encoded (Parent Education as 0, 1, 2; Gender as 0/1)
   - Features standardized for Linear Regression
3. Exploratory data analysis: marks distribution, study hours vs marks, attendance vs marks, marks by parent education and gender, correlation heatmap.
4. Model building: Linear Regression, Decision Tree (max depth 5), Random Forest (200 trees).
5. Training and testing: trained on 800 students, tested on 200 unseen students.
6. Evaluation: MAE, MSE, RMSE, R², plus a train vs test overfitting check.
7. Deployment: interactive Streamlit app on Streamlit Community Cloud.

##  Results (200 unseen test students)

| Model             | MAE | MSE | RMSE | R² |
| Linear Regression |3.36| 18.14 | 4.26 | 0.884 |
| Random Forest     | 3.76 | 22.76 | 4.77 | 0.854 |
| Decision Tree     | 4.82 | 35.88 | 5.99 | 0.771 |
| Baseline          | 10.15 | 157.64 | 12.56 | -0.008 | (predict the avg)

Overfitting check (R² on train vs test):

| Model | Train R² | Test R² | Gap |
|---|---|---|---|
| Linear Regression | 0.894 | 0.884 | 0.010 |
| Decision Tree | 0.869 | 0.771 | 0.099 |
| Random Forest | 0.981 | 0.855 | 0.127 |

## Key findings

- Linear Regression performed best: predictions are off by about 3.4 marks on average, and the model explains about 88% of the differences between students' marks.
- It generalizes well, with almost no gap between training and test scores.
- It beat Random Forest because the relationship in this dataset is mostly linear. The best model depends on the data, and a simple model is not a weak model.
- Study hours was among the strongest factors, while gender had almost no effect.

##  The App

Enter a student's details in the sidebar and the app shows:

- Predicted final marks and a likely range (prediction ± the model's test MAE)
- A support message: on track, moderate risk, or may need extra support
- A "Why this prediction?" chart showing how many marks each factor adds or removes compared with an average student
- An About tab with the model's test metrics

> The support thresholds (below 50 and below 65) are example values for the demo, not official rules.

##  Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit, Google Colab, VS Code, Git and GitHub


##  What I Learned

- Split the data "before" filling missing values or scaling, so no test information leaks into training.
- Compare models against a "baseline" and a train/test gap, not just a single score.
- A simple model can beat a complex one when the pattern is simple.
- Explaining "why" a model predicts something is as valuable as the prediction itself.
- Deployment has its own challenges, such as library versions and file paths.

##  Limitations

- The data is synthetic, so real-world accuracy would differ.
- Predictions are a guide for support decisions, not a final judgment about a student.

*Made by Uzma Kainat*
