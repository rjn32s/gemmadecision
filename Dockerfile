# CPU deployment. The published model is downloaded on first startup, never
# embedded in the image. Use a GPU image/runtime for CUDA or vLLM deployments.
FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.14.0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY python/ ./python/

# Installing the CPU wheel first keeps this CPU image free of CUDA libraries.
RUN python -m pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install . \
    && useradd --create-home --uid 10001 --user-group gemmadecision \
    && mkdir -p /models \
    && chown 10001:10001 /models

USER 10001:10001
VOLUME ["/models"]
EXPOSE 8700
HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8700/ready', timeout=4)"

# Supply GEMMADECISION_API_KEY at runtime when listening on 0.0.0.0.
ENTRYPOINT ["gemmadecision"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8700", "--device", "cpu"]
