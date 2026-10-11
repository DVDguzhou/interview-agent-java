# 国内网络可传 --build-arg BASE_REGISTRY=docker.m.daocloud.io/library/
ARG BASE_REGISTRY=
# ML 服务镜像（CPU 版：简历解析；评分模型推理需 GPU 镜像 + --extra train）
FROM ${BASE_REGISTRY}python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra serve --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra serve

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "iml-serve"]
