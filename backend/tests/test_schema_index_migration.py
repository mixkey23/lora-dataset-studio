"""Indexes on a database that already existed.

`index=True` on a model column is only honoured when db.create_all() CREATES the
table. On an install that predates the column, the additive migration adds the
column and stops there — so every one of those installs has been scanning the
table without the index ever since, and nothing in the codebase ever emitted a
CREATE INDEX to catch up.
"""


def _indexes(db, table):
    from sqlalchemy import text
    return {row[1] for row in db.session.execute(text(f'PRAGMA index_list({table})'))}


def test_a_missing_index_is_created_on_an_existing_database(app):
    """The legacy shape, reproduced: drop the index the way a database that never
    had it looks, then boot the migration and require it back."""
    from sqlalchemy import text
    from app import _INDEX_ADDITIONS, _apply_additive_migrations
    from app.extensions import db
    with app.app_context():
        for table, col in _INDEX_ADDITIONS:
            db.session.execute(text(f'DROP INDEX IF EXISTS ix_{table}_{col}'))
        db.session.commit()
        for table, col in _INDEX_ADDITIONS:
            assert f'ix_{table}_{col}' not in _indexes(db, table)

        _apply_additive_migrations()

        for table, col in _INDEX_ADDITIONS:
            assert f'ix_{table}_{col}' in _indexes(db, table), f'{table}.{col}'


def test_applying_the_migration_twice_is_a_no_op(app):
    """It runs on every boot, so it has to be idempotent and never raise."""
    from app import _INDEX_ADDITIONS, _apply_additive_migrations
    from app.extensions import db
    with app.app_context():
        _apply_additive_migrations()
        _apply_additive_migrations()
        for table, col in _INDEX_ADDITIONS:
            assert f'ix_{table}_{col}' in _indexes(db, table)


def test_the_index_list_matches_the_models(app):
    """Keeps the tuple honest: every entry must name a real column that the model
    declares index=True, so a renamed column fails here instead of silently
    creating an index on nothing (or on the wrong thing)."""
    from app import _INDEX_ADDITIONS, _SCHEMA_ADDITIONS
    from app.extensions import db
    tables = db.metadata.tables
    added = {(t, c) for t, c, _ in _SCHEMA_ADDITIONS}
    for table, col in _INDEX_ADDITIONS:
        column = tables[table].columns[col]
        assert column.index is True, f'{table}.{col} is not declared index=True'
        # …and it must be one of the columns the additive path adds — a column
        # that has always existed already got its index at create time.
        assert (table, col) in added, f'{table}.{col} is not an additive column'
