#!/bin/sh
# Container entrypoint: prepare the database and static files, then run the
# command given to `docker run` / the image CMD (gunicorn by default).
set -eu

python manage.py migrate --noinput
python manage.py collectstatic --noinput

exec "$@"
