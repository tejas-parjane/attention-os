import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.database import (
    Base,
    Event,
    User,
    get_engine,
    get_session,
    init_db,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DB_PATH = Path(__file__).resolve().parent.parent / "attention_os.db"


def load_events() -> pd.DataFrame:
    parquet_path = DATA_DIR / "events.parquet"
    logger.info("Loading events from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d events", len(df))
    return df


def load_user_metadata() -> pd.DataFrame:
    csv_path = DATA_DIR / "user_metadata.csv"
    if not csv_path.exists():
        csv_path = DATA_DIR / "users.csv"
    logger.info("Loading user metadata from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d users", len(df))
    return df


def seed_users(session, users_df: pd.DataFrame) -> None:
    existing = {row[0] for row in session.execute(select(User.user_id)).all()}
    inserted = 0
    for _, row in users_df.iterrows():
        uid = str(row["user_id"])
        if uid in existing:
            continue
        user = User(
            user_id=uid,
            archetype=str(row.get("archetype", "unknown")),
            signup_date=pd.to_datetime(row.get("signup_date", pd.Timestamp.now())),
            country=str(row.get("country", "unknown")),
            device_type=str(row.get("device_type", "unknown")),
            acquisition_channel=str(row.get("acquisition_channel", "unknown")),
        )
        session.add(user)
        inserted += 1
    session.commit()
    logger.info("Inserted %d new users (%d already existed)", inserted, len(existing))


def seed_events(session, events_df: pd.DataFrame) -> None:
    user_ids = {row[0] for row in session.execute(select(User.user_id)).all()}
    inserted = 0
    skipped = 0
    for _, row in events_df.iterrows():
        uid = str(row["user_id"])
        if uid not in user_ids:
            skipped += 1
            continue
        event = Event(
            user_id=uid,
            session_id=str(row.get("session_id", "")),
            timestamp=pd.to_datetime(row.get("timestamp", pd.Timestamp.now())),
            event_type=str(row.get("event_type", "unknown")),
            session_duration=_safe_float(row.get("session_duration")),
            screens_viewed=_safe_int(row.get("screens_viewed")),
            actions_completed=_safe_int(row.get("actions_completed")),
            level=_safe_int(row.get("level")),
            purchase_amount=float(row.get("purchase_amount", 0.0) or 0.0),
            ad_impressions=int(row.get("ad_impressions", 0) or 0),
            ad_revenue=float(row.get("ad_revenue", 0.0) or 0.0),
            notification_opened=bool(row.get("notification_opened", False)),
            notification_clicked=bool(row.get("notification_clicked", False)),
            content_category=_safe_str(row.get("content_category")),
            device_type=_safe_str(row.get("device_type")),
            country=_safe_str(row.get("country")),
            acquisition_channel=_safe_str(row.get("acquisition_channel")),
        )
        session.add(event)
        inserted += 1
        if inserted % 10000 == 0:
            session.commit()
            logger.info("  ... committed %d events so far", inserted)
    session.commit()
    logger.info("Inserted %d events (%d skipped due to missing user)", inserted, skipped)


def _safe_float(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return float(val)


def _safe_int(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return int(val)


def _safe_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val)


def print_summary(engine) -> None:
    session = get_session(engine)
    try:
        user_count = session.query(User).count()
        event_count = session.query(Event).count()
        unique_event_users = session.query(Event.user_id).distinct().count()
        logger.info("=== Database Summary ===")
        logger.info("Users:              %d", user_count)
        logger.info("Events:             %d", event_count)
        logger.info("Unique event users: %d", unique_event_users)
        logger.info("DB file:            %s", DB_PATH)
    finally:
        session.close()


def main() -> None:
    logger.info("Starting database seed")
    users_df = load_user_metadata()
    events_df = load_events()

    engine = get_engine(f"sqlite:///{DB_PATH}")
    init_db(engine)

    session = get_session(engine)
    try:
        seed_users(session, users_df)
        seed_events(session, events_df)
    finally:
        session.close()

    print_summary(engine)
    logger.info("Seed complete")


if __name__ == "__main__":
    main()
