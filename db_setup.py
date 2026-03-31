import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = "hospital.db"


def create_connection():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    return conn, cursor


def create_tables(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        gender TEXT NOT NULL,
        birth_date TEXT NOT NULL,
        city TEXT NOT NULL,
        phone TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        department_id INTEGER PRIMARY KEY,
        department_name TEXT NOT NULL,
        floor_number INTEGER NOT NULL,
        building_name TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        specialty TEXT NOT NULL,
        department_id INTEGER NOT NULL,
        email TEXT NOT NULL,
        FOREIGN KEY (department_id) REFERENCES departments(department_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id INTEGER PRIMARY KEY,
        patient_id INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL,
        department_id INTEGER NOT NULL,
        appointment_date TEXT NOT NULL,
        reason TEXT NOT NULL,
        diagnosis TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id),
        FOREIGN KEY (department_id) REFERENCES departments(department_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prescriptions (
        prescription_id INTEGER PRIMARY KEY,
        appointment_id INTEGER NOT NULL,
        patient_id INTEGER NOT NULL,
        medication_name TEXT NOT NULL,
        dosage TEXT NOT NULL,
        duration_days INTEGER NOT NULL,
        FOREIGN KEY (appointment_id) REFERENCES appointments(appointment_id),
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    );
    """)


def clear_tables(cursor):
    # Respecter l'ordre inverse des dépendances
    cursor.execute("DELETE FROM prescriptions;")
    cursor.execute("DELETE FROM appointments;")
    cursor.execute("DELETE FROM doctors;")
    cursor.execute("DELETE FROM patients;")
    cursor.execute("DELETE FROM departments;")


def seed_departments(cursor):
    departments_data = [
        (1, "Cardiology", 2, "Building A"),
        (2, "Neurology", 3, "Building A"),
        (3, "Pediatrics", 1, "Building B"),
        (4, "Emergency", 0, "Main Building"),
        (5, "Dermatology", 2, "Building C"),
    ]

    cursor.executemany(
        """
    INSERT INTO departments (department_id, department_name, floor_number, building_name)
    VALUES (?, ?, ?, ?)
    """,
        departments_data,
    )


def seed_patients(cursor):
    first_names = [
        "Ali",
        "Lina",
        "Karim",
        "Sara",
        "John",
        "Emma",
        "Noah",
        "Lucas",
        "Maya",
        "Adam",
        "Yasmine",
        "Hugo",
        "Lea",
        "Omar",
        "Nina",
    ]
    last_names = [
        "Hassan",
        "Dupont",
        "Martin",
        "Benali",
        "Smith",
        "Lambert",
        "Leroy",
        "Moreau",
        "Diallo",
        "Bernard",
    ]
    cities = ["Brussels", "Mons", "Liege", "Namur", "Paris", "Charleroi"]
    genders = ["M", "F"]

    patients_data = []
    for i in range(1, 51):
        first = random.choice(first_names)
        last = random.choice(last_names)
        gender = random.choice(genders)
        year = random.randint(1960, 2010)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        birth_date = f"{year:04d}-{month:02d}-{day:02d}"
        city = random.choice(cities)
        phone = str(random.randint(100000, 999999))
        patients_data.append((i, first, last, gender, birth_date, city, phone))

    cursor.executemany(
        """
    INSERT INTO patients (patient_id, first_name, last_name, gender, birth_date, city, phone)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        patients_data,
    )


def seed_doctors(cursor):
    first_names = [
        "Sarah",
        "Ahmed",
        "Julie",
        "Thomas",
        "Sophie",
        "Yacine",
        "Laura",
        "David",
        "Nora",
        "Mehdi",
    ]
    last_names = [
        "Martin",
        "Benali",
        "Lambert",
        "Leroy",
        "Dupuis",
        "Hassan",
        "Moreau",
        "Petit",
        "Bernard",
        "Diallo",
    ]

    doctors_data = [
        (
            1,
            first_names[0],
            last_names[0],
            "Cardiologist",
            1,
            "sarah.martin@hospital.com",
        ),
        (
            2,
            first_names[1],
            last_names[1],
            "Neurologist",
            2,
            "ahmed.benali@hospital.com",
        ),
        (
            3,
            first_names[2],
            last_names[2],
            "Pediatrician",
            3,
            "julie.lambert@hospital.com",
        ),
        (
            4,
            first_names[3],
            last_names[3],
            "Emergency Physician",
            4,
            "thomas.leroy@hospital.com",
        ),
        (
            5,
            first_names[4],
            last_names[4],
            "Dermatologist",
            5,
            "sophie.dupuis@hospital.com",
        ),
        (
            6,
            first_names[5],
            last_names[5],
            "Cardiologist",
            1,
            "yacine.hassan@hospital.com",
        ),
        (
            7,
            first_names[6],
            last_names[6],
            "Neurologist",
            2,
            "laura.moreau@hospital.com",
        ),
        (
            8,
            first_names[7],
            last_names[7],
            "Pediatrician",
            3,
            "david.petit@hospital.com",
        ),
        (
            9,
            first_names[8],
            last_names[8],
            "Emergency Physician",
            4,
            "nora.bernard@hospital.com",
        ),
        (
            10,
            first_names[9],
            last_names[9],
            "Dermatologist",
            5,
            "mehdi.diallo@hospital.com",
        ),
    ]

    cursor.executemany(
        """
    INSERT INTO doctors (doctor_id, first_name, last_name, specialty, department_id, email)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        doctors_data,
    )


def seed_appointments(cursor):
    reasons_by_department = {
        1: ["chest pain", "high blood pressure", "palpitations", "cardiac checkup"],
        2: ["headache", "migraine", "dizziness", "memory issues"],
        3: ["fever", "cough", "routine child checkup", "vaccination follow-up"],
        4: ["accident", "injury", "acute pain", "trauma"],
        5: ["rash", "skin irritation", "acne", "allergy reaction"],
    }

    diagnoses_by_department = {
        1: ["hypertension", "arrhythmia", "angina", "stable condition"],
        2: ["migraine", "stress headache", "neuropathy", "observation needed"],
        3: ["viral infection", "seasonal flu", "routine monitoring", "mild infection"],
        4: ["injury", "fracture suspicion", "muscle trauma", "wound care"],
        5: ["dermatitis", "allergy", "eczema", "skin infection"],
    }

    statuses = ["completed", "completed", "completed", "scheduled", "cancelled"]

    appointments_data = []
    for i in range(1, 81):
        patient_id = random.randint(1, 50)
        doctor_id = random.randint(1, 10)

        # récupérer le department du médecin pour garder la cohérence
        department_id = ((doctor_id - 1) % 5) + 1

        date = datetime.now() - timedelta(days=random.randint(0, 365))
        date_str = date.strftime("%Y-%m-%d")

        reason = random.choice(reasons_by_department[department_id])
        diagnosis = random.choice(diagnoses_by_department[department_id])
        status = random.choice(statuses)

        appointments_data.append(
            (
                i,
                patient_id,
                doctor_id,
                department_id,
                date_str,
                reason,
                diagnosis,
                status,
            )
        )

    cursor.executemany(
        """
    INSERT INTO appointments (
        appointment_id, patient_id, doctor_id, department_id,
        appointment_date, reason, diagnosis, status
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        appointments_data,
    )


def seed_prescriptions(cursor):
    medications_by_department = {
        1: ["Aspirin", "Beta Blocker", "Atorvastatin", "Enalapril"],
        2: ["Ibuprofen", "Paracetamol", "Sumatriptan", "Vitamin B Complex"],
        3: ["Paracetamol", "Amoxicillin", "Cough Syrup", "Ibuprofen"],
        4: ["Ibuprofen", "Paracetamol", "Amoxicillin", "Diclofenac"],
        5: ["Hydrocortisone Cream", "Antihistamine", "Amoxicillin", "Paracetamol"],
    }

    dosages = ["100mg", "250mg", "400mg", "500mg", "50mg"]
    duration_choices = [3, 5, 7, 10, 14, 30]

    # on ne crée pas forcément une prescription pour chaque rendez-vous
    prescription_id = 1
    prescriptions_data = []

    # récupérer tous les rendez-vous complétés pour lier logiquement les prescriptions
    cursor.execute("""
    SELECT appointment_id, patient_id, department_id
    FROM appointments
    WHERE status = 'completed'
    """)
    completed_appointments = cursor.fetchall()

    for appointment_id, patient_id, department_id in completed_appointments:
        num_prescriptions = random.randint(0, 2)

        for _ in range(num_prescriptions):
            medication = random.choice(medications_by_department[department_id])
            dosage = random.choice(dosages)
            duration = random.choice(duration_choices)

            prescriptions_data.append(
                (
                    prescription_id,
                    appointment_id,
                    patient_id,
                    medication,
                    dosage,
                    duration,
                )
            )
            prescription_id += 1

    cursor.executemany(
        """
    INSERT INTO prescriptions (
        prescription_id, appointment_id, patient_id,
        medication_name, dosage, duration_days
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        prescriptions_data,
    )


def main():
    # Si tu veux repartir proprement à chaque exécution :
    # supprime l'ancien fichier avant recréation.
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()

    conn, cursor = create_connection()

    try:
        create_tables(cursor)
        clear_tables(cursor)
        seed_departments(cursor)
        seed_patients(cursor)
        seed_doctors(cursor)
        seed_appointments(cursor)
        seed_prescriptions(cursor)

        conn.commit()

        # petits contrôles
        cursor.execute("SELECT COUNT(*) FROM patients;")
        patients_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM doctors;")
        doctors_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM appointments;")
        appointments_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM prescriptions;")
        prescriptions_count = cursor.fetchone()[0]

        print("Base de données complète créée avec succès !")
        print(f"Patients: {patients_count}")
        print(f"Doctors: {doctors_count}")
        print(f"Appointments: {appointments_count}")
        print(f"Prescriptions: {prescriptions_count}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
