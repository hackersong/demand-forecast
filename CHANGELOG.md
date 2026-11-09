# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
简体中文版本：[CHANGELOG.zh-CN.md](CHANGELOG.zh-CN.md)

## [Unreleased]

### Added

* Real account management (`user_management`, issue #10): self-service
  registration with email validation and Django password validators, a role
  model on auth groups (admin / planner / viewer) enforced on every write
  operation, an admin-only user & role management page, a user dashboard and
  password change / reset flows. Ships with 29 page-level tests and updated
  translations in all six languages.
* Service layer for the forecasting engine (`model_training/services/`,
  issue #9): `algorithms.py` (the forecast algorithms), `metrics.py`,
  `trainer.py` (preprocessing and orchestration) and `constants.py`.
  `views.py` shrank from ~2500 to ~900 lines and keeps re-exports, so
  `apps.demand_prediction` imports are unchanged. A behaviour snapshot over
  the pure functions (metrics, Croston/SBA, trend, decomposition,
  preprocessing, retry/fallback tables) proved outputs identical before and
  after the extraction; 12 new unit tests exercise the services without the
  Django test client.
* Docker support (issue #7): `Dockerfile` (python:3.12-slim + gettext +
  gunicorn), `docker-compose.yml` with database/media volumes,
  `docker-entrypoint.sh` running migrate + collectstatic on start, WhiteNoise
  for static files under `DEBUG=False` and a CI job that builds the image and
  smoke-tests `/login/`.
* Multilingual user interface: English (default), 中文, Français, Deutsch,
  Español and 日本語, switchable at runtime from the top right selector.
* Translation catalogs for all six languages plus
  `scripts/sync_zh_catalog.py`, which keeps the Simplified Chinese catalog in
  sync with the source strings.
* Locale aware date rendering (`localize_date` / `localize_month` template
  filters) so months and dates follow the active language.
* `apps.core.context_processors.languages` exposes `AVAILABLE_LANGUAGES_NATIVE`
  so the switcher can list every language in its own language.
* Documentation set in English and Chinese: README, installation, usage,
  architecture and translations guides.
* Community health files: `LICENSE` (MIT), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates and a GitHub Actions
  workflow.
* `.env.example` and environment-driven configuration for `SECRET_KEY`,
  `DEBUG`, `ALLOWED_HOSTS` and `LANGUAGE_CODE`.
* Module screenshots under `docs/images/`, referenced from both READMEs and
  the usage guide (PR #11 by @abhinavkumarpathak1-coder).

### Changed

* The language switcher moved from the sidebar footer to the top right corner
  and now uses a high-contrast (amber) pill style.
* The account chip (username + log out) also moved from the sidebar footer to
  the top right bar.
* Dates in tables are rendered with the locale's date format instead of the
  language neutral `%Y-%m` / `%Y-%m-%d` output.
* Sidebar menu, page title and footer use translated strings as well, so the
  Simplified Chinese UI no longer shows English navigation entries.
* The default language is always `LANGUAGE_CODE` (`en`): the browser
  `Accept-Language` header is ignored unless `DJANGO_LANGUAGE_FROM_BROWSER` is
  enabled (`DefaultLanguageLocaleMiddleware`).
* `SECRET_KEY` is no longer hard-coded; it is read from `DJANGO_SECRET_KEY`
  (with an insecure development fallback).
* `DEBUG` and `ALLOWED_HOSTS` are no longer hard-coded `True` / `['*']`.
* `Content-Language` now follows the active language instead of always being
  `zh-CN`.
* `requirements.txt` uses minimum versions instead of exact pins and documents
  the optional algorithm dependencies.

### Fixed

* `model_results` and the training page raised `NameError` because the service
  layer extraction dropped `from .models import ForecastModel, ForecastResult`
  from `model_training/views.py`. Both pages are covered by tests now, and a
  static check (`apps/core/tests.py`) fails the suite when any view module
  references a name that no scope binds.
* UI text was rendered two to seven times in a row (for example the home page
  hint or the upload format description). The catalogs contained `msgstr` values
  that repeated the same sentence, because the translation was written both as a
  full sentence and again as wrapped continuation lines. All six catalogs were
  repaired (en 18, fr 27, de 21, es 26, ja 17 entries) and the Simplified
  Chinese catalog was regenerated.
* `scripts/sync_zh_catalog.py` no longer keeps the continuation lines of the
  previous `msgstr`, so re-syncing the Chinese catalog cannot reintroduce the
  duplication.
* `upload_data` returned HTTP 500 on every request path that called `gettext`:
  the row loop reused `_`, the gettext alias, as a throwaway variable, which
  made `_` a local variable and broke every `_()` call in the view
  (`UnboundLocalError`). Fixed in PR #11 by @abhinavkumarpathak1-coder by
  renaming the loop variable.

### Security

* `db.sqlite3`, `.env`, `__pycache__`, `media/`, `staticfiles/` and compiled
  `*.mo` files are excluded from version control.
