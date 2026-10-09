import sqlite3
import pandas as pd
import os

print("Loading Real Synthea CSVs into SQL Database...")

# Ensure processed directory exists
os.makedirs('data/processed', exist_ok=True)

# Load CSVs 
patients = pd.read_csv('data/raw/patients.csv', low_memory=False)
encounters = pd.read_csv('data/raw/encounters.csv', low_memory=False)

# Connect to SQLite
conn = sqlite3.connect('data/hospital.db')
patients.to_sql('patients', conn, if_exists='replace', index=False)
encounters.to_sql('encounters', conn, if_exists='replace', index=False)

# Write the Advanced SQL Query using Window Functions
sql_query = """
CREATE TABLE IF NOT EXISTS master_patient_flow AS
WITH Encounter_History AS (
    SELECT 
        e.Id AS Encounter_ID,
        e.PATIENT AS Patient_ID,
        p.GENDER AS Gender,
        e.ENCOUNTERCLASS AS Department,
        e.DESCRIPTION AS Encounter_Type,
        e.REASONDESCRIPTION AS Diagnosis,
        e.START AS Arrival_Time,
        e.STOP AS Discharge_Time,
        e.TOTAL_CLAIM_COST AS Claim_Cost,
        e.PAYER_COVERAGE AS Insurance_Coverage,
        -- Use Window Function to find the date of the patient's PREVIOUS visit
        LAG(e.STOP) OVER (PARTITION BY e.PATIENT ORDER BY e.START) AS Prev_Discharge_Time
    FROM encounters e
    JOIN patients p ON e.PATIENT = p.Id
    WHERE e.ENCOUNTERCLASS IN ('emergency', 'inpatient', 'urgentcare', 'ambulatory')
)
SELECT 
    *,
    -- Calculate days between last discharge and current arrival
    CAST(julianday(Arrival_Time) - julianday(Prev_Discharge_Time) AS INTEGER) AS Days_Since_Last_Visit
FROM Encounter_History;
"""

cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS master_patient_flow")
cursor.execute(sql_query)
conn.commit()
conn.close()

print("SQL Join complete! master_patient_flow table created with Readmission tracking.")