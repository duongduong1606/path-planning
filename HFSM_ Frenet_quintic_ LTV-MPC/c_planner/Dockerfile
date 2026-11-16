FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential binutils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "PyYAML>=6" \
    "xmlschema>=4,<5" \
    "pytest>=7" \
    "matplotlib>=3.8"

WORKDIR /bench
