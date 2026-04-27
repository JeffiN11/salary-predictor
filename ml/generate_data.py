"""
Synthetic salary dataset generator.
Generates realistic salary data based on experience, role, location, and skills.
Run: python ml/generate_data.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 5000

JOB_TITLES = {
    "Data Analyst": 65000,
    "Junior Developer": 60000,
    "Backend Developer": 85000,
    "Frontend Developer": 78000,
    "Full Stack Developer": 90000,
    "ML Engineer": 105000,
    "Data Scientist": 100000,
    "DevOps Engineer": 95000,
    "AI Engineer": 110000,
    "Software Engineer": 92000,
}

LOCATIONS = {
    "Toronto": 1.15,
    "Vancouver": 1.10,
    "Montreal": 0.95,
    "Ottawa": 1.05,
    "Calgary": 1.08,
    "Remote": 1.00,
}

EDUCATION = {
    "High School": 0.85,
    "Bachelor": 1.00,
    "Master": 1.15,
    "PhD": 1.25,
}

SKILLS = ["Python", "SQL", "AWS", "Docker", "Kubernetes", "React", "FastAPI", "TensorFlow", "Spark", "Go"]


def generate_dataset():
    records = []
    for _ in range(N):
        title = np.random.choice(list(JOB_TITLES.keys()))
        location = np.random.choice(list(LOCATIONS.keys()))
        education = np.random.choice(list(EDUCATION.keys()))
        experience = np.random.randint(0, 20)
        num_skills = np.random.randint(1, 6)
        selected_skills = np.random.choice(SKILLS, num_skills, replace=False)

        base = JOB_TITLES[title]
        salary = (
            base
            * LOCATIONS[location]
            * EDUCATION[education]
            * (1 + experience * 0.03)
            + np.random.normal(0, 5000)
        )
        salary = max(40000, round(salary, -2))

        record = {
            "job_title": title,
            "location": location,
            "education": education,
            "years_experience": experience,
            "num_skills": num_skills,
            "salary": salary,
        }
        for skill in SKILLS:
            record[f"skill_{skill.lower()}"] = 1 if skill in selected_skills else 0

        records.append(record)

    df = pd.DataFrame(records)
    os.makedirs("ml/data", exist_ok=True)
    df.to_csv("ml/data/salaries.csv", index=False)
    print(f"Dataset generated: {len(df)} records -> ml/data/salaries.csv")
    print(df.describe())
    return df


if __name__ == "__main__":
    generate_dataset()
