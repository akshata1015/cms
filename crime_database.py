"""SQLite storage for crime alerts and simple dashboard analytics."""

import csv
import os
import sqlite3
from datetime import datetime


class CrimeDatabase:
    def __init__(self, path="crime_records.db"):
        self.path = path
        self._create_tables()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _create_tables(self):
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    location TEXT NOT NULL DEFAULT 'Local Camera',
                    source TEXT NOT NULL DEFAULT 'webcam'
                )"""
            )
            connection.commit()

    def add_alert(self, alert_type, details, location="Local Camera", source="webcam", timestamp=None):
        timestamp = timestamp or datetime.now().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO alerts(timestamp, alert_type, details, location, source) VALUES (?, ?, ?, ?, ?)",
                (timestamp, alert_type, details, location, source),
            )
            connection.commit()

    def recent(self, limit=50):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT timestamp, alert_type, details, location, source FROM alerts "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return rows

    def counts_by_type(self):
        with self._connect() as connection:
            return connection.execute(
                "SELECT alert_type, COUNT(*) FROM alerts GROUP BY alert_type ORDER BY COUNT(*) DESC"
            ).fetchall()

    def export_csv(self, path="crime_records_export.csv"):
        rows = self.recent(10000)
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["timestamp", "alert_type", "details", "location", "source"])
            writer.writerows(rows)
        return path


class AlertStore:
    """Keeps the legacy CSV log and writes the same event to SQLite."""

    def __init__(self, csv_path="alerts_log.csv", database_path="crime_records.db", location="Local Camera", source="webcam"):
        self.csv_path = csv_path
        self.database = CrimeDatabase(database_path)
        self.location = location
        self.source = source
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as output:
                csv.writer(output).writerow(["timestamp", "alert_type", "details"])

    def log(self, alert_type, details):
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow([timestamp, alert_type, details])
        self.database.add_alert(alert_type, details, self.location, self.source, timestamp)
