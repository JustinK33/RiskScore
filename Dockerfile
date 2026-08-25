# Serving image: the scoring API and the dashboard, from one process.
#
# Two stages so the runtime layer contains a built virtualenv and nothing that
# built it - no pip cache, no compilers, no pyproject.toml, no tests. The venv
# is copied wholesale rather than installed again, which is why the runtime
# stage needs no package index at all.
#
# 3.13, not 3.14: every pin in requirements-serve.txt ships a cp313 manylinux
# wheel, so this image never compiles anything. On a brand-new minor version
# one missing wheel turns a 30-second build into a numpy-from-source build that
# needs a toolchain this image does not have.
#
# Build and run:
#
#     docker build -t riskscore .
#     docker run --rm -p 8000:8000 -v "$PWD/reports:/app/reports:ro" riskscore
#
# The reports tree is a mount, not a layer. A model bundle is generated output
# that changes every retrain, and baking one in would make the image the thing
# you rebuild to roll a model back. `riskscore activate` is that thing.

FROM python:3.13-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /src
RUN python -m venv /opt/venv

# Dependencies before source, in their own layer: the pins change on a dependency
# bump and the source changes on every commit, so this ordering means an ordinary
# code change reuses the whole install.
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# `--no-deps` because the line above already resolved everything. Without it pip
# would re-resolve from the floors in pyproject.toml and could quietly upgrade
# past a pin, which would defeat the point of having pinned.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .


FROM python:3.13-slim AS runtime

# No build tools, no pip, no git in this stage - only what the venv needs to run.
# Nothing in [serve] links a C library outside libc, so there is no apt-get here
# at all: no libgomp1, because xgboost is the [train] extra and is not installed.
# The cost is real and worth stating: this image cannot load an xgboost bundle,
# because unpickling one imports xgboost. Serving a compared XGBoost run means
# adding xgboost to requirements-serve.txt and libgomp1 to a
# `RUN apt-get install` here, at roughly double the image size. The default
# bundle is logistic regression.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root, with a fixed high uid so a bind-mounted reports tree has a stable
# owner to grant read access to. A scoring service needs no write access to
# anything, so root buys nothing and costs a container escape.
RUN useradd --create-home --uid 10001 riskscore

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY dashboard ./dashboard

# Created and chowned before the volume exists, because a Docker named volume
# inherits the ownership of the directory it covers. Without this, a
# `-v riskscore-reports:/app/reports` mount lands root-owned and unreadable.
RUN mkdir -p /app/reports && chown riskscore:riskscore /app/reports

USER riskscore

# 0.0.0.0 plus ALLOW_PUBLIC_BIND, because a container that binds loopback is a
# container nothing can reach. That pair is deliberate rather than a default -
# see the settings validator - and it is safe here only because the mutating
# routes stay off: ALLOW_UPLOAD and ALLOW_RETRAIN are unset, so an unauthenticated
# remote fit is not reachable no matter how the container is published.
#
# REQUIRE_BUNDLE=1 inverts the local default. A fresh clone should boot and say
# "train something"; a container that cannot score should fail immediately and be
# replaced, rather than answer /healthz while every /predict returns 503.
#
# LOG_JSON=1 because the audience is a log aggregator, not a terminal.
#
# ALLOWED_HOSTS is left at its default (localhost, 127.0.0.1) so `docker run -p
# 8000:8000` works from a browser. Behind a proxy or a real hostname, set
# RISKSCORE_ALLOWED_HOSTS to that hostname - the Host allow-list is what stops
# DNS rebinding, and `*` gives it up.
ENV RISKSCORE_HOST=0.0.0.0 \
    RISKSCORE_ALLOW_PUBLIC_BIND=1 \
    RISKSCORE_PORT=8000 \
    RISKSCORE_REPORTS_DIR=/app/reports \
    RISKSCORE_DASHBOARD_DIR=/app/dashboard \
    RISKSCORE_REQUIRE_BUNDLE=1 \
    RISKSCORE_LOG_JSON=1

EXPOSE 8000

# /readyz, not /healthz, and the body is parsed rather than trusted: a 200 from
# /healthz only means the process is alive, and a process serving no bundle is
# alive and useless. `bundle_loaded` is the condition worth restarting on.
#
# Python rather than curl, so the image needs no extra package, and it reads
# RISKSCORE_PORT itself so overriding the port does not silently break the check.
# start-period covers the bundle load and the SHAP background build.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json,os,sys,urllib.request; port=os.environ.get('RISKSCORE_PORT','8000'); body=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/readyz', timeout=4)); sys.exit(0 if body.get('bundle_loaded') else 1)"]

# The console script, not `python -m uvicorn`: `riskscore serve` is what the
# runbook documents, it resolves settings through the same validator every other
# entry point uses, and it configures logging before uvicorn can install its own
# dictConfig over the top.
#
# Exec form, so riskscore is PID 1 and receives SIGTERM directly. Under a shell
# it would be a child, the shell would ignore the signal, and every stop would
# take the full 10-second timeout and then a SIGKILL.
CMD ["riskscore", "serve"]
