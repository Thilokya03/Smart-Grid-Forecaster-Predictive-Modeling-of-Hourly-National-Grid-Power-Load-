#!/bin/sh
set -eu

APP_DIR="${APP_DIR:-/app}"
STORAGE_DIR="${STORAGE_DIR:-/app/storage}"

if [ "${RENDER:-}" = "true" ]; then
  mkdir -p "$STORAGE_DIR/data" "$STORAGE_DIR/artifacts" /input/demand

  for name in data artifacts; do
    target="$STORAGE_DIR/$name"
    link="$APP_DIR/$name"

    mkdir -p "$target"

    if [ -L "$link" ]; then
      rm "$link"
    elif [ -d "$link" ]; then
      if [ "$(ls -A "$link" 2>/dev/null)" ]; then
        cp -a "$link/." "$target/"
      fi
      rm -rf "$link"
    elif [ -e "$link" ]; then
      rm -f "$link"
    fi

    ln -s "$target" "$link"
  done
fi

exec python -m ui.pipeline_dashboard
