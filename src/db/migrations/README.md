# PostgreSQL Migrations

Alembic migrations for the PostgreSQL source-of-truth schema live in
`src/db/migrations/versions/`.

The 004 schema is intentionally split by implementation slice instead of a
single `0001` file:

- `40be5b7bbb1d_postgresql_source_of_truth_us1.py`: core research ledger
- `d305f4d1ff77_postgresql_source_of_truth_us2.py`: projection and graph records
- `07c7492119e4_postgresql_source_of_truth_us3.py`: local/demo UX records
- `56464a69bd55_add_query_indexes.py`: query indexes
- `8f1a2b3c4d5e_provider_contract_records.py`: 006 provider contract extension

Run migrations with `alembic upgrade head` after setting `DATABASE_URL` or
`TEST_DATABASE_URL`.
