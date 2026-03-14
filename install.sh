#!/bin/bash
# Python 환경 패키지 설치 스크립트 (Python 3.10 기준)
# 사용법: bash install.sh

set -e

PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"

echo "=== [1/2] 일반 패키지 설치 (torch CUDA 인덱스 포함) ==="
pip install -r requirements.txt --extra-index-url "$PYTORCH_INDEX"

echo ""
echo "=== [2/2] 의존성 충돌 패키지 설치 (--no-deps) ==="
pip install --no-deps -r requirements-nodeps.txt

echo ""
echo "설치 완료."
