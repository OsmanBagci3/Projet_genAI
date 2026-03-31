$contextDir = ".\context"

if (!(Test-Path $contextDir)) {
    New-Item -ItemType Directory -Path $contextDir | Out-Null
}

$schema = @"
DATABASE: hospital

TABLE patients
- patient_id (INTEGER, PRIMARY KEY)
- first_name (TEXT)
- last_name (TEXT)
- gender (TEXT)
- birth_date (TEXT)
- city (TEXT)
- phone (TEXT)

TABLE departments
- department_id (INTEGER, PRIMARY KEY)
- department_name (TEXT)
- floor_number (INTEGER)
- building_name (TEXT)

TABLE doctors
- doctor_id (INTEGER, PRIMARY KEY)
- first_name (TEXT)
- last_name (TEXT)
- specialty (TEXT)
- department_id (INTEGER, FOREIGN KEY -> departments.department_id)
- email (TEXT)

TABLE appointments
- appointment_id (INTEGER, PRIMARY KEY)
- patient_id (INTEGER, FOREIGN KEY -> patients.patient_id)
- doctor_id (INTEGER, FOREIGN KEY -> doctors.doctor_id)
- department_id (INTEGER, FOREIGN KEY -> departments.department_id)
- appointment_date (TEXT)
- reason (TEXT)
- diagnosis (TEXT)
- status (TEXT)

TABLE prescriptions
- prescription_id (INTEGER, PRIMARY KEY)
- appointment_id (INTEGER, FOREIGN KEY -> appointments.appointment_id)
- patient_id (INTEGER, FOREIGN KEY -> patients.patient_id)
- medication_name (TEXT)
- dosage (TEXT)
- duration_days (INTEGER)

RELATIONSHIPS
- patients -> appointments : one-to-many
- doctors -> appointments : one-to-many
- departments -> doctors : one-to-many
- departments -> appointments : one-to-many
- appointments -> prescriptions : one-to-many
- patients -> prescriptions : one-to-many
"@

$tableDescriptions = @"
TABLE patients
This table stores patient demographic information.
Each row represents one patient.
Useful columns:
- patient_id: unique identifier of a patient
- first_name, last_name: patient name
- gender: patient gender
- birth_date: patient birth date
- city: city where the patient lives
- phone: contact number

TABLE departments
This table stores hospital departments or medical services.
Each row represents one department.
Useful columns:
- department_id: unique identifier of a department
- department_name: name of the department, such as Cardiology or Neurology
- floor_number: floor where the department is located
- building_name: hospital building name

TABLE doctors
This table stores doctor information.
Each doctor belongs to exactly one department.
Useful columns:
- doctor_id: unique identifier of a doctor
- first_name, last_name: doctor name
- specialty: medical specialty
- department_id: link to the department
- email: doctor email

TABLE appointments
This table stores consultations or medical appointments.
Each appointment links one patient, one doctor, and one department.
Useful columns:
- appointment_id: unique identifier of an appointment
- patient_id: link to the patient
- doctor_id: link to the doctor
- department_id: link to the department
- appointment_date: date of the appointment
- reason: why the patient came
- diagnosis: doctor diagnosis
- status: appointment status such as completed, scheduled, or cancelled

TABLE prescriptions
This table stores prescribed medications.
Each prescription is linked to one appointment and one patient.
Useful columns:
- prescription_id: unique identifier of a prescription
- appointment_id: link to the appointment
- patient_id: link to the patient
- medication_name: prescribed medicine name
- dosage: prescribed dosage
- duration_days: treatment duration in days
"@

$vocabulary = @"
patient -> patients
patients -> patients
malade -> patients
personne -> patients

doctor -> doctors
doctors -> doctors
médecin -> doctors
médecins -> doctors
specialist -> doctors
specialty -> doctors.specialty
spécialité -> doctors.specialty

department -> departments
departments -> departments
service -> departments
services -> departments
unité -> departments
floor -> departments.floor_number
building -> departments.building_name

appointment -> appointments
appointments -> appointments
consultation -> appointments
consultations -> appointments
rendez-vous -> appointments
visit -> appointments
visits -> appointments
reason -> appointments.reason
motif -> appointments.reason
diagnosis -> appointments.diagnosis
diagnostic -> appointments.diagnosis
status -> appointments.status
date -> appointments.appointment_date

prescription -> prescriptions
prescriptions -> prescriptions
ordonnance -> prescriptions
ordonnances -> prescriptions
medication -> prescriptions.medication_name
médicament -> prescriptions.medication_name
medicine -> prescriptions.medication_name
dosage -> prescriptions.dosage
duration -> prescriptions.duration_days
durée -> prescriptions.duration_days

cardiology -> departments.department_name
neurology -> departments.department_name
pediatrics -> departments.department_name
emergency -> departments.department_name
dermatology -> departments.department_name
"@

$examples = @"
QUESTION: How many patients are in the database?
SQL:
SELECT COUNT(*) AS total_patients
FROM patients;

QUESTION: Which doctors work in Cardiology?
SQL:
SELECT d.first_name, d.last_name
FROM doctors d
JOIN departments dep ON d.department_id = dep.department_id
WHERE dep.department_name = 'Cardiology';

QUESTION: How many appointments are in the database?
SQL:
SELECT COUNT(*) AS total_appointments
FROM appointments;

QUESTION: Which patients live in Brussels?
SQL:
SELECT first_name, last_name
FROM patients
WHERE city = 'Brussels';

QUESTION: Which appointments were completed?
SQL:
SELECT appointment_id, appointment_date, reason, diagnosis
FROM appointments
WHERE status = 'completed';

QUESTION: Which medications were prescribed?
SQL:
SELECT DISTINCT medication_name
FROM prescriptions;

QUESTION: Which patients received Paracetamol?
SQL:
SELECT DISTINCT p.first_name, p.last_name
FROM patients p
JOIN prescriptions pr ON p.patient_id = pr.patient_id
WHERE pr.medication_name = 'Paracetamol';

QUESTION: How many appointments did each doctor have?
SQL:
SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments
FROM doctors d
LEFT JOIN appointments a ON d.doctor_id = a.doctor_id
GROUP BY d.doctor_id, d.first_name, d.last_name;

QUESTION: How many appointments took place in each department?
SQL:
SELECT dep.department_name, COUNT(a.appointment_id) AS total_appointments
FROM departments dep
LEFT JOIN appointments a ON dep.department_id = a.department_id
GROUP BY dep.department_id, dep.department_name;

QUESTION: Which doctor had the most appointments?
SQL:
SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments
FROM doctors d
JOIN appointments a ON d.doctor_id = a.doctor_id
GROUP BY d.doctor_id, d.first_name, d.last_name
ORDER BY total_appointments DESC
LIMIT 1;

QUESTION: Which patients have more than one prescription?
SQL:
SELECT p.first_name, p.last_name, COUNT(pr.prescription_id) AS total_prescriptions
FROM patients p
JOIN prescriptions pr ON p.patient_id = pr.patient_id
GROUP BY p.patient_id, p.first_name, p.last_name
HAVING COUNT(pr.prescription_id) > 1;

QUESTION: Show appointments with patient and doctor names.
SQL:
SELECT
    a.appointment_id,
    p.first_name AS patient_first_name,
    p.last_name AS patient_last_name,
    d.first_name AS doctor_first_name,
    d.last_name AS doctor_last_name,
    a.appointment_date,
    a.reason
FROM appointments a
JOIN patients p ON a.patient_id = p.patient_id
JOIN doctors d ON a.doctor_id = d.doctor_id;

QUESTION: Show appointments in Cardiology.
SQL:
SELECT p.first_name, p.last_name, a.appointment_date, a.reason
FROM appointments a
JOIN patients p ON a.patient_id = p.patient_id
JOIN departments dep ON a.department_id = dep.department_id
WHERE dep.department_name = 'Cardiology';

QUESTION: Count prescriptions by medication.
SQL:
SELECT medication_name, COUNT(*) AS total_prescriptions
FROM prescriptions
GROUP BY medication_name;

QUESTION: Show doctors with their department names.
SQL:
SELECT d.first_name, d.last_name, d.specialty, dep.department_name
FROM doctors d
JOIN departments dep ON d.department_id = dep.department_id;
"@

$sqlRules = @"
SQL GENERATION RULES

1. Generate only SQLite-compatible SQL.
2. Generate only SELECT queries.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE statements.
4. Use only tables and columns that exist in the provided schema.
5. Use explicit JOIN conditions when multiple tables are needed.
6. Prefer readable SQL with table aliases when useful.
7. If the user asks for counts, use COUNT(*).
8. If the user asks for unique values, use DISTINCT when needed.
9. If the question refers to a department such as Cardiology, filter using departments.department_name.
10. If the question refers to medication, use prescriptions.medication_name.
11. If the question refers to consultations or appointments, use the appointments table.
12. If the user asks something outside the database scope, do not invent SQL.
13. If the schema is insufficient to answer, return no SQL and explain the limitation.
14. When grouping results, use GROUP BY correctly.
15. Keep the SQL concise and correct.
"@

$joinPatterns = @"
JOIN PATTERNS

PATTERN 1
patients -> appointments
JOIN appointments a ON patients.patient_id = a.patient_id

PATTERN 2
doctors -> appointments
JOIN appointments a ON doctors.doctor_id = a.doctor_id

PATTERN 3
departments -> doctors
JOIN doctors d ON departments.department_id = d.department_id

PATTERN 4
departments -> appointments
JOIN appointments a ON departments.department_id = a.department_id

PATTERN 5
appointments -> prescriptions
JOIN prescriptions pr ON appointments.appointment_id = pr.appointment_id

PATTERN 6
patients -> prescriptions
JOIN prescriptions pr ON patients.patient_id = pr.patient_id

PATTERN 7
appointments -> patients -> doctors
FROM appointments a
JOIN patients p ON a.patient_id = p.patient_id
JOIN doctors d ON a.doctor_id = d.doctor_id

PATTERN 8
appointments -> departments
FROM appointments a
JOIN departments dep ON a.department_id = dep.department_id

PATTERN 9
doctors -> departments -> appointments
FROM doctors d
JOIN departments dep ON d.department_id = dep.department_id
JOIN appointments a ON d.doctor_id = a.doctor_id

PATTERN 10
patients -> prescriptions -> appointments
FROM patients p
JOIN prescriptions pr ON p.patient_id = pr.patient_id
JOIN appointments a ON pr.appointment_id = a.appointment_id
"@

Set-Content -Path ".\context\schema.txt" -Value $schema -Encoding UTF8
Set-Content -Path ".\context\table_descriptions.txt" -Value $tableDescriptions -Encoding UTF8
Set-Content -Path ".\context\vocabulary.txt" -Value $vocabulary -Encoding UTF8
Set-Content -Path ".\context\examples.txt" -Value $examples -Encoding UTF8
Set-Content -Path ".\context\sql_rules.txt" -Value $sqlRules -Encoding UTF8
Set-Content -Path ".\context\join_patterns.txt" -Value $joinPatterns -Encoding UTF8

Write-Host "Dossier context créé avec succès."
Write-Host "Fichiers générés :"
Write-Host "- schema.txt"
Write-Host "- table_descriptions.txt"
Write-Host "- vocabulary.txt"
Write-Host "- examples.txt"
Write-Host "- sql_rules.txt"
Write-Host "- join_patterns.txt"