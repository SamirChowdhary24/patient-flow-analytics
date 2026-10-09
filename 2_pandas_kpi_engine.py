import sqlite3
import pandas as pd
import numpy as np
import os

print("Connecting to database and extracting table...")

# 1. Connect to SQLite database and pull the joined table
conn = sqlite3.connect('data/hospital.db')
df = pd.read_sql_query("SELECT * FROM master_patient_flow", conn)
conn.close()

# 2. Parse timestamps
df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'], format='ISO8601', errors='coerce')
df['Discharge_Time'] = pd.to_datetime(df['Discharge_Time'], format='ISO8601', errors='coerce')

# 3. Operational KPIs: Length of Stay (LOS) in Hours
df['Length_Of_Stay_Hours'] = (df['Discharge_Time'] - df['Arrival_Time']).dt.total_seconds() / 3600
df = df[df['Length_Of_Stay_Hours'] >= 0]  # Filter out erroneous records

# 4. Financial KPIs
df['Out_Of_Pocket_Cost'] = df['Claim_Cost'] - df['Insurance_Coverage']
df['Coverage_Ratio'] = np.where(df['Claim_Cost'] > 0, df['Insurance_Coverage'] / df['Claim_Cost'], 1.0)

# 5. Clinical KPI: 30-Day Readmission Flag
# A patient is a readmission if Days_Since_Last_Visit is between 0 and 30
df['Is_30_Day_Readmission'] = np.where(
    (df['Days_Since_Last_Visit'].notnull()) & 
    (df['Days_Since_Last_Visit'] >= 0) & 
    (df['Days_Since_Last_Visit'] <= 30), 
    1, 
    0
)

# 6. Statistical Anomaly Detection (IQR Method for Department Bottlenecks)
Q1 = df.groupby('Department')['Length_Of_Stay_Hours'].transform(lambda x: x.quantile(0.25))
Q3 = df.groupby('Department')['Length_Of_Stay_Hours'].transform(lambda x: x.quantile(0.75))
IQR = Q3 - Q1
df['Stat_Outlier_LOS'] = np.where(df['Length_Of_Stay_Hours'] > (Q3 + 1.5 * IQR), 'Severe Bottleneck', 'Normal')

# 7. Format timestamps for Power BI compatibility
df['Arrival_Time'] = df['Arrival_Time'].dt.tz_localize(None)
df['Discharge_Time'] = df['Discharge_Time'].dt.tz_localize(None)

# 8. Export processed dataset
os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/powerbi_ready_advanced.csv'
df.to_csv(output_path, index=False)

print(f"Success! Processed {len(df):,} records and saved to {output_path}")