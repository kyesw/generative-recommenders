#!/bin/bash
set -e

case "$1" in
    train)
        exec train
        ;;
    serve)
        exec uvicorn \
            --host 0.0.0.0 \
            --port 8080 \
            --workers 1 \
            --log-level info \
            --app-dir /opt/program \
            sagemaker_handler:app
        ;;
    *)
        exec "$@"
        ;;
esac
