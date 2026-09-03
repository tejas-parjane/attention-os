"""
Generate synthetic events + user metadata, run validation, and save outputs
to data/raw as events.parquet and users.csv.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the src package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.generator import generate_events, generate_user_metadata  # noqa: E402
from data.validation import DataValidationError, validate_events  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    n_users = 500
    days = 90
    seed = 42

    logger.info("Generating %d users over %d days (seed=%d)", n_users, days, seed)
    events_df = generate_events(n_users=n_users, days=days, seed=seed)
    users_df = generate_user_metadata(events_df)

    logger.info("Validating events DataFrame…")
    try:
        result = validate_events(events_df, fail_loud=True)
    except DataValidationError as exc:
        logger.error("Validation failed loudly:\n%s", exc)
        raise
    else:
        failed = [c for c in result.checks if c["status"] == "failed"]
        logger.info(
            "Validation passed=%s — %d check(s) failed",
            result.passed,
            len(failed),
        )
        if failed:
            for c in failed:
                logger.warning("  [%s] %s", c["name"], c["message"])

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "events.parquet"
    csv_path = raw_dir / "users.csv"

    events_df.to_parquet(parquet_path, index=False)
    users_df.to_csv(csv_path, index=False)

    logger.info("Saved events → %s (%d rows)", parquet_path, len(events_df))
    logger.info("Saved users  → %s (%d rows)", csv_path, len(users_df))

    print("\n=== Summary ===")
    print(f"Events  : {len(events_df):,} rows")
    print(f"Users   : {len(users_df):,}")
    print(f"Date range: {events_df['timestamp'].min()} -> {events_df['timestamp'].max()}")
    print("\nValidation checks:")
    for c in result.checks:
        print(f"  [{c['status']:<6}] {c['name']:<20} {c['message']}")
    print("\nEvent type breakdown:")
    print(events_df["event_type"].value_counts().to_string())
    print("\nArchetype distribution (metadata):")
    print(users_df["archetype_label"].value_counts().to_string())
    print("\nDevice distribution:")
    print(events_df["device_type"].value_counts().to_string())
    print("\nCountry distribution:")
    print(events_df["country"].value_counts().to_string())


if __name__ == "__main__":
    main()
