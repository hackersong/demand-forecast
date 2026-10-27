# syntax=docker/dockerfile:1

# ---------- runtime image ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=config.settings \
    DJANGO_SECRET_KEY=unsafe-development-key-change-me \
    DJANGO_DEBUG=False \
    DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1 \
    DJANGO_DB_PATH=/app/data/db.sqlite3 \
    PYTHONPATH=/app

# gettext provides msgfmt for `manage.py compilemessages`, so the UI can be
# translated; build-essential is needed for the wheels that have no manylinux
# build (scikit-learn / xgboost fall back to a local build on some platforms).
RUN apt-get update \
    && apt-get install -y --no-install-recommends gettext \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so that source changes do not invalidate the layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# `data/` holds the SQLite file, `staticfiles/` the collected assets. Both are
# owned by the runtime user so the container also starts without a volume.
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /app/staticfiles /app/media \
    && chown -R appuser:appuser /app \
    && chmod +x /app/docker-entrypoint.sh

USER appuser

RUN python manage.py compilemessages \
    && python manage.py collectstatic --noinput

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/login/')"

ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "config.wsgi:application"]
