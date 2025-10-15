# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Create the dataset
# -----------------------------
data = {
    'study_hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'exam_score': [50, 55, 60, 62, 65, 70, 75, 78, 85, 90],
    'sleeping_hours': [9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Compute Covariance & Correlation between two attributes
# -----------------------------
cov_study_exam = np.cov(df['study_hours'], df['exam_score'])[0][1]
corr_study_exam = np.corrcoef(df['study_hours'], df['exam_score'])[0][1]

print("Covariance between study_hours and exam_score:", cov_study_exam)
print("Correlation between study_hours and exam_score:", corr_study_exam)

# -----------------------------
# 3. Compute Covariance Matrix
# -----------------------------
cov_matrix = df.cov()
print("\nCovariance Matrix:")
print(cov_matrix)

# -----------------------------
# 4. Compute Correlation Matrix
# -----------------------------
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# -----------------------------
# 5. Visualize Study Hours vs Exam Score
# -----------------------------
sns.lmplot(x='study_hours', y='exam_score', data=df)
plt.title('Study Hours vs Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.tight_layout()  # Ensures labels and title fit well
plt.show()
