#!/bin/bash
set -e

case "$1" in
    train)
        exec train
        ;;
    serve)
        exec gunicorn \
            --workers 1 \
            --worker-class sync \
            --bind 0.0.0.0:8080 \
            --timeout 300 \
            --preload \
            --log-level info \
            --chdir /opt/program \
            sagemaker_handler:app
        ;;
    *)
        exec "$@"
        ;;
esac
