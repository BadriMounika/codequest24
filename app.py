from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('student_performance_dataset.csv')

# Separate handling of missing values for numerical and categorical columns
# Fill missing numerical values with the column mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing categorical values with the most frequent value (mode)
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Convert categorical variables to numerical form
df['Contributions'] = df['Contributions'].map({'Mentor': 2, 'Course Assistant': 1, 'None': 0})
df['Club Activities'] = df['Club Activities'].map({'Leader': 2, 'Active Member': 1, 'Participant': 0, 'None': 0})

# Define features and target variable
features = ['CGPA', 'Semester 1 GPA', 'Semester 2 GPA', 'Semester 3 GPA', 
            'Semester 4 GPA', 'Semester 5 GPA', 'Semester 6 GPA', 'Semester 7 GPA',
            'Semester 8 GPA', 'Attendance', 'Hackathon Participation', 
            'Paper Presentations', 'Workshops and Seminars', 'Internships', 'Projects']
target = 'Overall Performance Score'

# Train the model
model = LinearRegression()
model.fit(df[features], df[target])

# Predict scores and rank students
df['Predicted Score'] = model.predict(df[features])
df['Rank'] = df['Predicted Score'].rank(ascending=False)

# Get the top 3 students for each batch
top_students = df.groupby('Batch', group_keys=False).apply(lambda x: x.nlargest(3, 'Predicted Score')).reset_index(drop=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html', students=top_students.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
