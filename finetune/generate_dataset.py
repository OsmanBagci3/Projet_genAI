"""
Génère automatiquement le dataset de fine-tuning NLQ->SQL
basé sur le schéma hospitalier réel.
Produit 3 fichiers JSONL: train / val / test
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# ── Schéma injecté dans chaque exemple ───────────────────────────────────────
SCHEMA = (
    "patients(patient_id, first_name, last_name, gender, birth_date, city, phone)\n"
    "doctors(doctor_id, first_name, last_name, specialty, department_id, email)\n"
    "appointments(appointment_id, patient_id, doctor_id, department_id, "
    "appointment_date, reason, diagnosis, status)\n"
    "prescriptions(prescription_id, appointment_id, patient_id, "
    "medication_name, dosage, duration_days)\n"
    "departments(department_id, department_name, floor_number, building_name)"
)

RULES = (
    "- Use ONLY the tables and columns listed in the schema above\n"
    "- Generate SQLite SELECT only\n"
    "- Use aliases: p=patients, d=doctors, a=appointments, pr=prescriptions, dep=departments\n"
    "- NEVER invent columns (e.g. doctor_name, age, appointment_time do NOT exist)\n"
    "- Always qualify columns with their alias (e.g. p.first_name, not first_name)"
)

INSTRUCTION = "Generate a valid SQLite SELECT query using only the provided schema."


def make(question: str, sql: str, category: str) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": f"Question: {question}\nSchema:\n{SCHEMA}\nRules:\n{RULES}",
        "output": sql,
        "category": category,
    }


# ── Dataset complet ───────────────────────────────────────────────────────────
examples = [

    # ════════════════════════════════
    #  COUNT SIMPLE
    # ════════════════════════════════
    make("How many patients are in the database?",
         "SELECT COUNT(*) AS total_patients FROM patients;", "count"),
    make("Count the total number of patients.",
         "SELECT COUNT(*) AS total_patients FROM patients;", "count"),
    make("Give me the patient count.",
         "SELECT COUNT(*) AS total_patients FROM patients;", "count"),
    make("Total patients registered?",
         "SELECT COUNT(*) AS total_patients FROM patients;", "count"),
    make("How many people are registered as patients?",
         "SELECT COUNT(*) AS total_patients FROM patients;", "count"),
    make("How many appointments are in the database?",
         "SELECT COUNT(*) AS total_appointments FROM appointments;", "count"),
    make("Count all appointments.",
         "SELECT COUNT(*) AS total_appointments FROM appointments;", "count"),
    make("Total number of appointments?",
         "SELECT COUNT(*) AS total_appointments FROM appointments;", "count"),
    make("How many consultations are recorded?",
         "SELECT COUNT(*) AS total_appointments FROM appointments;", "count"),
    make("Give me the appointment count.",
         "SELECT COUNT(*) AS total_appointments FROM appointments;", "count"),
    make("How many doctors are in the hospital?",
         "SELECT COUNT(*) AS total_doctors FROM doctors;", "count"),
    make("Count all doctors.",
         "SELECT COUNT(*) AS total_doctors FROM doctors;", "count"),
    make("Total number of doctors?",
         "SELECT COUNT(*) AS total_doctors FROM doctors;", "count"),
    make("How many physicians are registered?",
         "SELECT COUNT(*) AS total_doctors FROM doctors;", "count"),
    make("Give me the doctor count.",
         "SELECT COUNT(*) AS total_doctors FROM doctors;", "count"),
    make("How many departments exist?",
         "SELECT COUNT(*) AS total_departments FROM departments;", "count"),
    make("Count all departments.",
         "SELECT COUNT(*) AS total_departments FROM departments;", "count"),
    make("Total number of departments?",
         "SELECT COUNT(*) AS total_departments FROM departments;", "count"),
    make("How many services are in the hospital?",
         "SELECT COUNT(*) AS total_departments FROM departments;", "count"),
    make("How many prescriptions are recorded?",
         "SELECT COUNT(*) AS total_prescriptions FROM prescriptions;", "count"),
    make("Count all prescriptions.",
         "SELECT COUNT(*) AS total_prescriptions FROM prescriptions;", "count"),
    make("Total number of prescriptions?",
         "SELECT COUNT(*) AS total_prescriptions FROM prescriptions;", "count"),
    make("Give me the prescription count.",
         "SELECT COUNT(*) AS total_prescriptions FROM prescriptions;", "count"),

    # ════════════════════════════════
    #  FILTER
    # ════════════════════════════════
    make("Which appointments were completed?",
         "SELECT a.appointment_id, a.appointment_date, a.reason, a.diagnosis "
         "FROM appointments a WHERE a.status = 'completed' ORDER BY a.appointment_date DESC;", "filter"),
    make("Show all completed appointments.",
         "SELECT a.appointment_id, a.appointment_date, a.reason, a.diagnosis "
         "FROM appointments a WHERE a.status = 'completed' ORDER BY a.appointment_date DESC;", "filter"),
    make("List appointments with status completed.",
         "SELECT a.appointment_id, a.appointment_date, a.reason, a.diagnosis "
         "FROM appointments a WHERE a.status = 'completed' ORDER BY a.appointment_date DESC;", "filter"),
    make("Find completed consultations.",
         "SELECT a.appointment_id, a.appointment_date, a.reason, a.diagnosis "
         "FROM appointments a WHERE a.status = 'completed' ORDER BY a.appointment_date DESC;", "filter"),
    make("Display finished appointments.",
         "SELECT a.appointment_id, a.appointment_date, a.reason, a.diagnosis "
         "FROM appointments a WHERE a.status = 'completed' ORDER BY a.appointment_date DESC;", "filter"),
    make("Which appointments were cancelled?",
         "SELECT a.appointment_id, a.appointment_date, a.reason "
         "FROM appointments a WHERE a.status = 'cancelled' ORDER BY a.appointment_date DESC;", "filter"),
    make("Show all cancelled appointments.",
         "SELECT a.appointment_id, a.appointment_date, a.reason "
         "FROM appointments a WHERE a.status = 'cancelled' ORDER BY a.appointment_date DESC;", "filter"),
    make("List appointments with status cancelled.",
         "SELECT a.appointment_id, a.appointment_date, a.reason "
         "FROM appointments a WHERE a.status = 'cancelled' ORDER BY a.appointment_date DESC;", "filter"),
    make("Which patients are male?",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.gender = 'M' ORDER BY p.last_name;", "filter"),
    make("Show female patients.",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.gender = 'F' ORDER BY p.last_name;", "filter"),
    make("List patients from Brussels.",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.city = 'Brussels' ORDER BY p.last_name;", "filter"),
    make("Show patients from Paris.",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.city = 'Paris' ORDER BY p.last_name;", "filter"),
    make("Which patients live in Liege?",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.city = 'Liege' ORDER BY p.last_name;", "filter"),
    make("Find patients from Namur.",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.city = 'Namur' ORDER BY p.last_name;", "filter"),
    make("Show patients from Mons.",
         "SELECT p.first_name, p.last_name FROM patients p WHERE p.city = 'Mons' ORDER BY p.last_name;", "filter"),
    make("Which doctors are cardiologists?",
         "SELECT d.first_name, d.last_name FROM doctors d WHERE d.specialty = 'Cardiologist' ORDER BY d.last_name;", "filter"),
    make("Show neurologists.",
         "SELECT d.first_name, d.last_name FROM doctors d WHERE d.specialty = 'Neurologist' ORDER BY d.last_name;", "filter"),
    make("List pediatricians.",
         "SELECT d.first_name, d.last_name FROM doctors d WHERE d.specialty = 'Pediatrician' ORDER BY d.last_name;", "filter"),
    make("Find dermatologists.",
         "SELECT d.first_name, d.last_name FROM doctors d WHERE d.specialty = 'Dermatologist' ORDER BY d.last_name;", "filter"),
    make("Show emergency physicians.",
         "SELECT d.first_name, d.last_name FROM doctors d WHERE d.specialty = 'Emergency Physician' ORDER BY d.last_name;", "filter"),
    make("List scheduled appointments.",
         "SELECT a.appointment_id, a.appointment_date, a.reason "
         "FROM appointments a WHERE a.status = 'scheduled' ORDER BY a.appointment_date;", "filter"),
    make("Show all appointments that are scheduled.",
         "SELECT a.appointment_id, a.appointment_date, a.reason "
         "FROM appointments a WHERE a.status = 'scheduled' ORDER BY a.appointment_date;", "filter"),

    # ════════════════════════════════
    #  JOIN
    # ════════════════════════════════
    make("Which patients received Paracetamol?",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Paracetamol' ORDER BY p.last_name;", "join"),
    make("List patients prescribed Paracetamol.",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Paracetamol' ORDER BY p.last_name;", "join"),
    make("Find patients who took Paracetamol.",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Paracetamol' ORDER BY p.last_name;", "join"),
    make("Which patients received Ibuprofen?",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Ibuprofen' ORDER BY p.last_name;", "join"),
    make("List patients prescribed Aspirin.",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Aspirin' ORDER BY p.last_name;", "join"),
    make("Which patients received Amoxicillin?",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Amoxicillin' ORDER BY p.last_name;", "join"),
    make("Find patients prescribed Beta Blocker.",
         "SELECT DISTINCT p.first_name, p.last_name FROM patients p "
         "JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "WHERE pr.medication_name = 'Beta Blocker' ORDER BY p.last_name;", "join"),
    make("Which doctors work in Cardiology?",
         "SELECT d.first_name, d.last_name FROM doctors d "
         "JOIN departments dep ON d.department_id = dep.department_id "
         "WHERE dep.department_name = 'Cardiology' ORDER BY d.last_name;", "join"),
    make("List doctors in Neurology.",
         "SELECT d.first_name, d.last_name FROM doctors d "
         "JOIN departments dep ON d.department_id = dep.department_id "
         "WHERE dep.department_name = 'Neurology' ORDER BY d.last_name;", "join"),
    make("Show doctors in Pediatrics.",
         "SELECT d.first_name, d.last_name FROM doctors d "
         "JOIN departments dep ON d.department_id = dep.department_id "
         "WHERE dep.department_name = 'Pediatrics' ORDER BY d.last_name;", "join"),
    make("Find doctors in Emergency.",
         "SELECT d.first_name, d.last_name FROM doctors d "
         "JOIN departments dep ON d.department_id = dep.department_id "
         "WHERE dep.department_name = 'Emergency' ORDER BY d.last_name;", "join"),
    make("Which doctors work in Dermatology?",
         "SELECT d.first_name, d.last_name FROM doctors d "
         "JOIN departments dep ON d.department_id = dep.department_id "
         "WHERE dep.department_name = 'Dermatology' ORDER BY d.last_name;", "join"),
    make("Show appointments in Cardiology.",
         "SELECT p.first_name, p.last_name, a.appointment_date, a.reason "
         "FROM appointments a "
         "JOIN patients p ON a.patient_id = p.patient_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Cardiology' ORDER BY a.appointment_date DESC;", "join"),
    make("List appointments in Neurology.",
         "SELECT p.first_name, p.last_name, a.appointment_date, a.reason "
         "FROM appointments a "
         "JOIN patients p ON a.patient_id = p.patient_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Neurology' ORDER BY a.appointment_date DESC;", "join"),
    make("Find appointments in Emergency.",
         "SELECT p.first_name, p.last_name, a.appointment_date, a.reason "
         "FROM appointments a "
         "JOIN patients p ON a.patient_id = p.patient_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Emergency' ORDER BY a.appointment_date DESC;", "join"),
    make("Show appointments in Dermatology.",
         "SELECT p.first_name, p.last_name, a.appointment_date, a.reason "
         "FROM appointments a "
         "JOIN patients p ON a.patient_id = p.patient_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Dermatology' ORDER BY a.appointment_date DESC;", "join"),
    make("Show appointments in Pediatrics.",
         "SELECT p.first_name, p.last_name, a.appointment_date, a.reason "
         "FROM appointments a "
         "JOIN patients p ON a.patient_id = p.patient_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Pediatrics' ORDER BY a.appointment_date DESC;", "join"),

    # ════════════════════════════════
    #  AGGREGATION
    # ════════════════════════════════
    make("How many appointments did each doctor have?",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d LEFT JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name ORDER BY total_appointments DESC;", "aggregation"),
    make("Count appointments per doctor.",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d LEFT JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name ORDER BY total_appointments DESC;", "aggregation"),
    make("Show the number of appointments for each doctor.",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d LEFT JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name ORDER BY total_appointments DESC;", "aggregation"),
    make("How many appointments per department?",
         "SELECT dep.department_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM departments dep LEFT JOIN appointments a ON dep.department_id = a.department_id "
         "GROUP BY dep.department_id, dep.department_name ORDER BY total_appointments DESC;", "aggregation"),
    make("Count appointments in each department.",
         "SELECT dep.department_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM departments dep LEFT JOIN appointments a ON dep.department_id = a.department_id "
         "GROUP BY dep.department_id, dep.department_name ORDER BY total_appointments DESC;", "aggregation"),
    make("Count prescriptions by medication.",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "aggregation"),
    make("How many times was each medication prescribed?",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "aggregation"),
    make("Show prescription count per medication.",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "aggregation"),
    make("Which medications were prescribed most?",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "aggregation"),
    make("How many patients per city?",
         "SELECT p.city, COUNT(p.patient_id) AS total_patients "
         "FROM patients p GROUP BY p.city ORDER BY total_patients DESC;", "aggregation"),
    make("Count patients by city.",
         "SELECT p.city, COUNT(p.patient_id) AS total_patients "
         "FROM patients p GROUP BY p.city ORDER BY total_patients DESC;", "aggregation"),
    make("Show number of patients in each city.",
         "SELECT p.city, COUNT(p.patient_id) AS total_patients "
         "FROM patients p GROUP BY p.city ORDER BY total_patients DESC;", "aggregation"),
    make("How many male and female patients are there?",
         "SELECT p.gender, COUNT(*) AS total "
         "FROM patients p GROUP BY p.gender ORDER BY total DESC;", "aggregation"),
    make("Count patients by gender.",
         "SELECT p.gender, COUNT(*) AS total "
         "FROM patients p GROUP BY p.gender ORDER BY total DESC;", "aggregation"),

    # ════════════════════════════════
    #  TOP-K / RANKING
    # ════════════════════════════════
    make("Which doctor had the most appointments?",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name "
         "ORDER BY total_appointments DESC LIMIT 1;", "topk"),
    make("Who is the busiest doctor?",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name "
         "ORDER BY total_appointments DESC LIMIT 1;", "topk"),
    make("Find the doctor with the highest number of appointments.",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name "
         "ORDER BY total_appointments DESC LIMIT 1;", "topk"),
    make("List the top 5 doctors by number of appointments.",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name "
         "ORDER BY total_appointments DESC LIMIT 5;", "topk"),
    make("Show the 5 most active doctors.",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total_appointments "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name "
         "ORDER BY total_appointments DESC LIMIT 5;", "topk"),
    make("What are the 5 most recent appointments?",
         "SELECT a.appointment_date, p.first_name, p.last_name, a.reason "
         "FROM appointments a JOIN patients p ON a.patient_id = p.patient_id "
         "ORDER BY a.appointment_date DESC LIMIT 5;", "topk"),
    make("Show the last 5 appointments.",
         "SELECT a.appointment_date, p.first_name, p.last_name, a.reason "
         "FROM appointments a JOIN patients p ON a.patient_id = p.patient_id "
         "ORDER BY a.appointment_date DESC LIMIT 5;", "topk"),
    make("Which city has the most patients with appointments?",
         "SELECT p.city, COUNT(DISTINCT p.patient_id) AS total_patients "
         "FROM patients p JOIN appointments a ON p.patient_id = a.patient_id "
         "GROUP BY p.city ORDER BY total_patients DESC LIMIT 1;", "topk"),
    make("Find the city with the highest number of patients.",
         "SELECT p.city, COUNT(DISTINCT p.patient_id) AS total_patients "
         "FROM patients p JOIN appointments a ON p.patient_id = a.patient_id "
         "GROUP BY p.city ORDER BY total_patients DESC LIMIT 1;", "topk"),
    make("What is the most prescribed medication?",
         "SELECT pr.medication_name, COUNT(*) AS total "
         "FROM prescriptions pr GROUP BY pr.medication_name ORDER BY total DESC LIMIT 1;", "topk"),
    make("Which medication is prescribed most often?",
         "SELECT pr.medication_name, COUNT(*) AS total "
         "FROM prescriptions pr GROUP BY pr.medication_name ORDER BY total DESC LIMIT 1;", "topk"),

    # ════════════════════════════════
    #  HAVING / COMPLEXE
    # ════════════════════════════════
    make("Which patients have more than one prescription?",
         "SELECT p.first_name, p.last_name, COUNT(pr.prescription_id) AS total_prescriptions "
         "FROM patients p JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "GROUP BY p.patient_id, p.first_name, p.last_name "
         "HAVING COUNT(pr.prescription_id) > 1 ORDER BY total_prescriptions DESC;", "having"),
    make("List patients with at least 2 prescriptions.",
         "SELECT p.first_name, p.last_name, COUNT(pr.prescription_id) AS total_prescriptions "
         "FROM patients p JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "GROUP BY p.patient_id, p.first_name, p.last_name "
         "HAVING COUNT(pr.prescription_id) > 1 ORDER BY total_prescriptions DESC;", "having"),
    make("Find patients prescribed more than once.",
         "SELECT p.first_name, p.last_name, COUNT(pr.prescription_id) AS total_prescriptions "
         "FROM patients p JOIN prescriptions pr ON p.patient_id = pr.patient_id "
         "GROUP BY p.patient_id, p.first_name, p.last_name "
         "HAVING COUNT(pr.prescription_id) > 1 ORDER BY total_prescriptions DESC;", "having"),
    make("Show patients who visited more than one department.",
         "SELECT p.first_name, p.last_name, COUNT(DISTINCT a.department_id) AS total_departments "
         "FROM patients p JOIN appointments a ON p.patient_id = a.patient_id "
         "GROUP BY p.patient_id, p.first_name, p.last_name "
         "HAVING COUNT(DISTINCT a.department_id) > 1 ORDER BY total_departments DESC;", "having"),
    make("Which patients consulted multiple departments?",
         "SELECT p.first_name, p.last_name, COUNT(DISTINCT a.department_id) AS total_departments "
         "FROM patients p JOIN appointments a ON p.patient_id = a.patient_id "
         "GROUP BY p.patient_id, p.first_name, p.last_name "
         "HAVING COUNT(DISTINCT a.department_id) > 1 ORDER BY total_departments DESC;", "having"),
    make("Find doctors with more than 5 appointments.",
         "SELECT d.first_name, d.last_name, COUNT(a.appointment_id) AS total "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name "
         "HAVING COUNT(a.appointment_id) > 5 ORDER BY total DESC;", "having"),
    make("Which cities have more than 5 patients?",
         "SELECT p.city, COUNT(*) AS total FROM patients p "
         "GROUP BY p.city HAVING COUNT(*) > 5 ORDER BY total DESC;", "having"),

    # ════════════════════════════════
    #  SEMANTIC MAPPING
    # ════════════════════════════════
    make("Which medications were prescribed?",
         "SELECT DISTINCT pr.medication_name FROM prescriptions pr ORDER BY pr.medication_name;", "semantic"),
    make("List all drugs prescribed to patients.",
         "SELECT DISTINCT pr.medication_name FROM prescriptions pr ORDER BY pr.medication_name;", "semantic"),
    make("Show available treatments in prescriptions.",
         "SELECT DISTINCT pr.medication_name FROM prescriptions pr ORDER BY pr.medication_name;", "semantic"),
    make("What is the most common disease in Emergency?",
         "SELECT a.diagnosis, COUNT(*) AS total_cases "
         "FROM appointments a JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Emergency' "
         "GROUP BY a.diagnosis ORDER BY total_cases DESC LIMIT 1;", "semantic"),
    make("Which diagnosis appears most in Emergency?",
         "SELECT a.diagnosis, COUNT(*) AS total_cases "
         "FROM appointments a JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Emergency' "
         "GROUP BY a.diagnosis ORDER BY total_cases DESC LIMIT 1;", "semantic"),
    make("What illness is most frequent in Cardiology?",
         "SELECT a.diagnosis, COUNT(*) AS total_cases "
         "FROM appointments a JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Cardiology' "
         "GROUP BY a.diagnosis ORDER BY total_cases DESC LIMIT 1;", "semantic"),
    make("Most common reason for visiting Neurology?",
         "SELECT a.reason, COUNT(*) AS total "
         "FROM appointments a JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Neurology' "
         "GROUP BY a.reason ORDER BY total DESC LIMIT 1;", "semantic"),
    make("Show departments ranked by completed appointments.",
         "SELECT dep.department_name, COUNT(a.appointment_id) AS total_completed "
         "FROM departments dep JOIN appointments a ON dep.department_id = a.department_id "
         "WHERE a.status = 'completed' "
         "GROUP BY dep.department_id, dep.department_name ORDER BY total_completed DESC;", "semantic"),
    make("Which service has the most finished consultations?",
         "SELECT dep.department_name, COUNT(a.appointment_id) AS total_completed "
         "FROM departments dep JOIN appointments a ON dep.department_id = a.department_id "
         "WHERE a.status = 'completed' "
         "GROUP BY dep.department_id, dep.department_name ORDER BY total_completed DESC LIMIT 1;", "semantic"),
    make("Which medications are most used in Neurology?",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr "
         "JOIN appointments a ON pr.appointment_id = a.appointment_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Neurology' "
         "GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "semantic"),
    make("What drugs are prescribed in Cardiology?",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr "
         "JOIN appointments a ON pr.appointment_id = a.appointment_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Cardiology' "
         "GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "semantic"),
    make("Find treatments given in Pediatrics.",
         "SELECT pr.medication_name, COUNT(*) AS total_prescriptions "
         "FROM prescriptions pr "
         "JOIN appointments a ON pr.appointment_id = a.appointment_id "
         "JOIN departments dep ON a.department_id = dep.department_id "
         "WHERE dep.department_name = 'Pediatrics' "
         "GROUP BY pr.medication_name ORDER BY total_prescriptions DESC;", "semantic"),

    # ════════════════════════════════
    #  COUNT DISTINCT
    # ════════════════════════════════
    make("For each department, show total appointments and distinct patients.",
         "SELECT dep.department_name, COUNT(a.appointment_id) AS total_appointments, "
         "COUNT(DISTINCT a.patient_id) AS total_patients "
         "FROM departments dep LEFT JOIN appointments a ON dep.department_id = a.department_id "
         "GROUP BY dep.department_id, dep.department_name ORDER BY total_appointments DESC;", "count_distinct"),
    make("Show distinct medications prescribed per department.",
         "SELECT dep.department_name, COUNT(DISTINCT pr.medication_name) AS distinct_medications "
         "FROM departments dep "
         "JOIN appointments a ON dep.department_id = a.department_id "
         "JOIN prescriptions pr ON a.appointment_id = pr.appointment_id "
         "GROUP BY dep.department_id, dep.department_name ORDER BY distinct_medications DESC;", "count_distinct"),
    make("How many distinct patients does each doctor have?",
         "SELECT d.first_name, d.last_name, COUNT(DISTINCT a.patient_id) AS distinct_patients "
         "FROM doctors d JOIN appointments a ON d.doctor_id = a.doctor_id "
         "GROUP BY d.doctor_id, d.first_name, d.last_name ORDER BY distinct_patients DESC;", "count_distinct"),
    make("Count distinct cities where patients come from.",
         "SELECT COUNT(DISTINCT p.city) AS total_cities FROM patients p;", "count_distinct"),
    make("How many distinct medications are prescribed?",
         "SELECT COUNT(DISTINCT pr.medication_name) AS distinct_medications FROM prescriptions pr;", "count_distinct"),
    make("Count unique patients per department.",
         "SELECT dep.department_name, COUNT(DISTINCT a.patient_id) AS unique_patients "
         "FROM departments dep LEFT JOIN appointments a ON dep.department_id = a.department_id "
         "GROUP BY dep.department_id, dep.department_name ORDER BY unique_patients DESC;", "count_distinct"),
]

# ── Shuffle reproductible ─────────────────────────────────────────────────────
random.seed(42)
random.shuffle(examples)

n = len(examples)
train_end = int(n * 0.75)
val_end   = int(n * 0.875)

splits = {
    "train": examples[:train_end],
    "val":   examples[train_end:val_end],
    "test":  examples[val_end:],
}

# ── Ecriture JSONL ────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)

for split_name, split_data in splits.items():
    path = out_dir / f"dataset_{split_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in split_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{split_name:5s}: {len(split_data):3d} exemples  →  {path}")

print(f"\nTotal: {n} exemples")
cats = {}
for ex in examples:
    cats[ex["category"]] = cats.get(ex["category"], 0) + 1
print("\nRépartition par catégorie:")
for cat, cnt in sorted(cats.items()):
    print(f"  {cat:<15} {cnt}")
