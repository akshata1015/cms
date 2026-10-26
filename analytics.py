"""Small analytics helpers used by the Streamlit dashboard."""

from collections import Counter
from datetime import datetime


def hotspot_summary(rows):
    """Return locations ranked by number of stored alerts."""
    counts = Counter(row[3] for row in rows if len(row) > 3)
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)


def hourly_summary(rows):
    counts = Counter()
    for row in rows:
        try:
            hour = datetime.fromisoformat(row[0]).hour
            counts[f"{hour:02d}:00"] += 1
        except (ValueError, TypeError, IndexError):
            continue
    return sorted(counts.items())


def crowd_status(person_count, crowd_limit):
    if person_count >= crowd_limit:
        return "CROWD ALERT", True
    return "Normal", False


def hotspot_risk(rows):
    """Rank locations using recent alert volume for a demo-friendly risk score."""
    counts = Counter(row[3] for row in rows if len(row) > 3)
    maximum = max(counts.values(), default=1)
    return sorted(
        ((location, count, round((count / maximum) * 100, 1)) for location, count in counts.items()),
        key=lambda item: item[1], reverse=True,
    )
