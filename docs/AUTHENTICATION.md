# RETINA NEXUS authentication

RETINA NEXUS uses the existing FastAPI, SQLAlchemy, bcrypt, and JWT stack. It does not use frontend-only or automatically successful authentication.

## Routes

- `/login`: sign in with a persisted user account.
- `/signup`: create a persisted prototype workspace account.
- `/forgot-password`: submit a privacy-preserving reset request.
- `/platform/*`: authenticated application routes.

The browser stores non-remembered sessions in `sessionStorage`. Selecting **Remember me** stores the access token in `localStorage`. Sign out clears both stores. Protected routes validate the token with `GET /api/v1/auth/me`; missing, invalid, expired, inactive-user, and API-rejected sessions return to `/login`.

## Backend endpoints

- `POST /api/v1/auth/login`
- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/forgot-password`
- `GET /api/v1/auth/me`

Passwords are bcrypt hashes and are never returned by the API. New passwords require 12–72 characters, uppercase and lowercase letters, a number, and a symbol. Email addresses are normalized before lookup or storage.

## Role boundary

The public form records a professional title such as ophthalmologist, screening operator, researcher, or administrator. Public registration always receives the least-privilege `healthcare_worker` authorization role. Selecting “Administrator” does not grant administrator permissions; elevated access requires a separate controlled administration process.

## Password-reset limitation

The prototype reset-request endpoint deliberately returns the same response for known and unknown email addresses. No SMTP or transactional-email provider is configured in this repository, so external reset-email delivery remains a deployment integration requirement. The endpoint does not expose account existence, plaintext passwords, or reset tokens.
