# Container notes

The root `docker-compose.yml` runs the API, frontend, PostgreSQL, and Redis
locally. Redis is included as the cache/task-infrastructure boundary; a Celery
or equivalent worker can be added without changing the HTTP contracts. The
compose file is a development reference, not a production security profile.

See [deployment architecture](../../docs/DEPLOYMENT_ARCHITECTURE.md) for
edge/clinic synchronization, low-bandwidth behavior, model version
management, and the production hardening checklist. See
[monitoring](../../docs/monitoring.md) for operations telemetry and the
drift-validation boundary.
