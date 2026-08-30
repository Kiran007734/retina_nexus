# Setup

Copy `backend/.env.example` to `backend/.env` and `frontend/.env.example` to `frontend/.env`; change `SECRET_KEY` before any shared deployment. Use SQLite for a quick local API run or `docker compose up --build` for the PostgreSQL and Redis-backed stack.

For development, the API creates missing tables on startup. Use Alembic for controlled schema changes. Keep model weights, uploaded images, `.env` files, and database files out of version control.
