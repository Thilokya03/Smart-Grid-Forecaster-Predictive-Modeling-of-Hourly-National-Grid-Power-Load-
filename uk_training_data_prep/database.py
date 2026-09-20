"""Database persistence for pipeline datasets.

CSV exports remain the pipeline compatibility interface. When DATABASE_URL is
configured, each canonical dataframe is also published as an atomic database
snapshot.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence
from uuid import uuid4

import pandas as pd


DATABASE_URL_ENV = "DATABASE_URL"
DATABASE_SCHEMA_ENV = "DATABASE_SCHEMA"
DEFAULT_SCHEMA = "weather_pipeline"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def database_url() -> str | None:
    value = os.getenv(DATABASE_URL_ENV, "").strip()
    return value or None


def database_enabled() -> bool:
    return database_url() is not None


def normalize_database_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def _validated_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid database {label}: {value!r}")
    return value


def _schema_for_url(url: str) -> str | None:
    if url.startswith("sqlite:"):
        return None
    return _validated_identifier(
        os.getenv(DATABASE_SCHEMA_ENV, DEFAULT_SCHEMA), "schema"
    )


def _qualified_name(preparer, table_name: str, schema: str | None) -> str:
    quoted_table = preparer.quote(table_name)
    if schema is None:
        return quoted_table
    return f"{preparer.quote_schema(schema)}.{quoted_table}"


def publish_dataframe(
    frame: pd.DataFrame,
    table_name: str,
    *,
    key_columns: Sequence[str] = ("timestamp",),
    url: str | None = None,
) -> bool:
    """Publish a dataframe as an atomic table snapshot.

    Returns False when no database is configured. A configured database error
    is raised so a hosted update cannot silently leave PostgreSQL stale.
    """

    configured_url = url or database_url()
    if configured_url is None:
        return False
    configured_url = normalize_database_url(configured_url)
    if frame.empty:
        raise ValueError(f"Refusing to publish empty dataset {table_name!r}.")

    try:
        from sqlalchemy import create_engine, inspect, text
    except ImportError as exc:
        raise RuntimeError(
            "Database publishing requires SQLAlchemy and psycopg. "
            "Install the project requirements."
        ) from exc

    table_name = _validated_identifier(table_name, "table")
    for column in frame.columns:
        _validated_identifier(str(column), "column")
    missing_keys = set(key_columns).difference(frame.columns)
    if missing_keys:
        raise ValueError(
            f"{table_name} is missing database key columns: {sorted(missing_keys)}"
        )
    if key_columns:
        null_key_count = int(frame[list(key_columns)].isna().any(axis=1).sum())
        if null_key_count:
            raise ValueError(
                f"Refusing to publish {table_name!r}: {null_key_count} rows have "
                f"null database keys {list(key_columns)}."
            )
    if key_columns and frame.duplicated(subset=list(key_columns)).any():
        duplicate_count = int(
            frame.duplicated(subset=list(key_columns), keep=False).sum()
        )
        raise ValueError(
            f"Refusing to publish {table_name!r}: {duplicate_count} rows have "
            f"duplicate database keys {list(key_columns)}."
        )

    schema = _schema_for_url(configured_url)
    staging_name = _validated_identifier(
        f"_staging_{table_name}_{uuid4().hex[:10]}", "table"
    )
    engine = create_engine(configured_url, pool_pre_ping=True)
    preparer = engine.dialect.identifier_preparer
    target = _qualified_name(preparer, table_name, schema)
    staging = _qualified_name(preparer, staging_name, schema)
    chunksize = max(1, min(1_000, 60_000 // len(frame.columns)))

    try:
        with engine.begin() as connection:
            if schema is not None:
                quoted_schema = preparer.quote_schema(schema)
                connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted_schema}"))

            frame.to_sql(
                staging_name,
                connection,
                schema=schema,
                if_exists="fail",
                index=False,
                chunksize=chunksize,
                method="multi",
            )

            if inspect(connection).has_table(table_name, schema=schema):
                connection.execute(text(f"DROP TABLE {target}"))

            connection.execute(
                text(
                    f"ALTER TABLE {staging} RENAME TO "
                    f"{preparer.quote(table_name)}"
                )
            )

            if key_columns:
                index_name = _validated_identifier(
                    f"uq_{table_name}_{'_'.join(key_columns)}", "index"
                )
                quoted_columns = ", ".join(preparer.quote(key) for key in key_columns)
                connection.execute(
                    text(
                        f"CREATE UNIQUE INDEX {preparer.quote(index_name)} "
                        f"ON {target} ({quoted_columns})"
                    )
                )

            _record_publication(
                connection,
                table_name=table_name,
                row_count=len(frame),
                schema=schema,
                preparer=preparer,
                text=text,
            )
    finally:
        engine.dispose()

    print(f"Published database table -> {schema + '.' if schema else ''}{table_name}")
    return True


def read_dataframe(
    table_name: str,
    *,
    filters: Mapping[str, object] | None = None,
    order_by: Sequence[str] = (),
    url: str | None = None,
) -> pd.DataFrame | None:
    """Read a pipeline table, or return None when no database is configured."""

    configured_url = url or database_url()
    if configured_url is None:
        return None
    configured_url = normalize_database_url(configured_url)

    try:
        from sqlalchemy import MetaData, Table, create_engine, select
    except ImportError as exc:
        raise RuntimeError(
            "Database reads require SQLAlchemy and psycopg. "
            "Install the project requirements."
        ) from exc

    table_name = _validated_identifier(table_name, "table")
    schema = _schema_for_url(configured_url)
    engine = create_engine(configured_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            table = Table(
                table_name,
                MetaData(),
                schema=schema,
                autoload_with=connection,
            )
            statement = select(table)
            for column_name, value in (filters or {}).items():
                column_name = _validated_identifier(column_name, "column")
                if column_name not in table.c:
                    raise ValueError(
                        f"{table_name} has no filter column {column_name!r}."
                    )
                statement = statement.where(table.c[column_name] == value)
            for column_name in order_by:
                column_name = _validated_identifier(column_name, "column")
                if column_name not in table.c:
                    raise ValueError(
                        f"{table_name} has no ordering column {column_name!r}."
                    )
                statement = statement.order_by(table.c[column_name])
            return pd.read_sql(statement, connection)
    finally:
        engine.dispose()


def database_status(
    table_names: Sequence[str], *, url: str | None = None
) -> dict[str, object]:
    """Check connectivity and return row counts for expected tables."""

    configured_url = url or database_url()
    if configured_url is None:
        return {"enabled": False, "connected": False, "tables": {}}
    configured_url = normalize_database_url(configured_url)

    try:
        from sqlalchemy import MetaData, Table, create_engine, func, inspect, select
    except ImportError as exc:
        raise RuntimeError(
            "Database checks require SQLAlchemy and psycopg. "
            "Install the project requirements."
        ) from exc

    schema = _schema_for_url(configured_url)
    engine = create_engine(configured_url, pool_pre_ping=True)
    counts: dict[str, int | None] = {}
    try:
        with engine.connect() as connection:
            inspector = inspect(connection)
            for raw_name in table_names:
                table_name = _validated_identifier(raw_name, "table")
                if not inspector.has_table(table_name, schema=schema):
                    counts[table_name] = None
                    continue
                table = Table(
                    table_name,
                    MetaData(),
                    schema=schema,
                    autoload_with=connection,
                )
                counts[table_name] = int(
                    connection.scalar(select(func.count()).select_from(table)) or 0
                )
    finally:
        engine.dispose()
    return {"enabled": True, "connected": True, "tables": counts}


def _record_publication(
    connection,
    *,
    table_name: str,
    row_count: int,
    schema: str | None,
    preparer,
    text,
) -> None:
    runs_table = _qualified_name(preparer, "pipeline_runs", schema)
    connection.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {runs_table} ("
            "id VARCHAR(32) PRIMARY KEY, "
            "dataset_name VARCHAR(128) NOT NULL, "
            "published_at TIMESTAMP NOT NULL, "
            "row_count BIGINT NOT NULL"
            ")"
        )
    )
    connection.execute(
        text(
            f"INSERT INTO {runs_table} "
            "(id, dataset_name, published_at, row_count) "
            "VALUES (:id, :dataset_name, :published_at, :row_count)"
        ),
        {
            "id": uuid4().hex,
            "dataset_name": table_name,
            "published_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "row_count": row_count,
        },
    )
