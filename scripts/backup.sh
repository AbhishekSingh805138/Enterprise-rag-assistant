#!/usr/bin/env bash
# Backup and restore for the production stack.
#
# The previous procedure tarred /app/chroma_db from inside the api
# container. Since the move to CHROMA_MODE=server the api container no
# longer mounts that path — vectors live in the `chroma` service's own
# volume — so that command succeeded while archiving an empty directory.
# Backups looked healthy and would have restored nothing.
#
# This script backs up the Docker volumes directly, which is where the
# data actually is, and `verify` proves a backup is restorable rather
# than merely present.
#
#   ./scripts/backup.sh backup  [dir]
#   ./scripts/backup.sh verify  <archive-dir>
#   ./scripts/backup.sh restore <archive-dir>
#
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}"

# What actually holds state. Kafka is deliberately excluded: it carries
# in-flight events, and every one of them is reconstructable from the
# document registry plus object storage.
VOLUMES=(
  "chroma_data"      # vector index — the expensive thing to rebuild
  "checkpoint_data"  # document registry, metrics, memory, event queue
  "minio_data"       # original uploaded bytes — required to re-index
)

log() { printf '  %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

volume_name() { echo "${PROJECT}_$1"; }

volume_exists() {
  docker volume inspect "$(volume_name "$1")" >/dev/null 2>&1
}

do_backup() {
  local out="${1:-backups/$(date +%Y%m%d-%H%M%S)}"
  mkdir -p "$out"
  log "backing up to $out"

  for vol in "${VOLUMES[@]}"; do
    if ! volume_exists "$vol"; then
      log "skip $vol (not present)"
      continue
    fi
    # A throwaway container is the only way to read a named volume; the
    # application containers are not guaranteed to mount all of them.
    docker run --rm \
      -v "$(volume_name "$vol"):/src:ro" \
      -v "$(cd "$out" && pwd):/dst" \
      alpine:3.20 tar czf "/dst/${vol}.tar.gz" -C /src . 2>/dev/null
    log "$(printf '%-16s %s' "$vol" "$(du -h "$out/${vol}.tar.gz" | cut -f1)")"
  done

  # Record what the backup should contain, so verify has something to
  # check against rather than just asserting the file is non-empty.
  local count
  count="$(document_count || echo -1)"
  cat > "$out/manifest.txt" <<EOF
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
compose_file=$COMPOSE_FILE
project=$PROJECT
chroma_documents=$count
volumes=${VOLUMES[*]}
EOF
  log "manifest: chroma_documents=$count"
  log "done"
}

document_count() {
  # Ask the running API rather than the volume, so the number reflects
  # what the system can actually serve.
  local port="${API_PORT:-8000}"
  curl -fsS "http://localhost:${port}/health" 2>/dev/null \
    | sed -n 's/.*"document_count":\([-0-9]*\).*/\1/p'
}

do_verify() {
  local dir="${1:?usage: backup.sh verify <archive-dir>}"
  [ -d "$dir" ] || die "no such directory: $dir"
  log "verifying $dir"

  local failed=0
  for vol in "${VOLUMES[@]}"; do
    local archive="$dir/${vol}.tar.gz"
    if [ ! -f "$archive" ]; then
      log "MISSING  $vol"; failed=1; continue
    fi
    # Listing the archive proves it is readable, not merely present.
    local entries
    entries="$(tar tzf "$archive" 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$entries" -lt 2 ]; then
      log "EMPTY    $vol ($entries entries) — this is the failure the old procedure hid"
      failed=1
    else
      log "$(printf 'ok       %-16s %s entries' "$vol" "$entries")"
    fi
  done

  if [ -f "$dir/manifest.txt" ]; then
    local expected
    expected="$(sed -n 's/^chroma_documents=//p' "$dir/manifest.txt")"
    log "manifest says chroma_documents=$expected"
    [ "${expected:-0}" -gt 0 ] 2>/dev/null || {
      log "WARNING  manifest records no documents; a restore would produce an empty index"
      failed=1
    }
  else
    log "WARNING  no manifest.txt"; failed=1
  fi

  [ "$failed" -eq 0 ] || die "verification FAILED — do not rely on this backup"
  log "verification passed"
}

do_restore() {
  local dir="${1:?usage: backup.sh restore <archive-dir>}"
  [ -d "$dir" ] || die "no such directory: $dir"
  do_verify "$dir"

  log "stopping services"
  docker compose -f "$COMPOSE_FILE" down

  for vol in "${VOLUMES[@]}"; do
    local archive="$dir/${vol}.tar.gz"
    [ -f "$archive" ] || continue
    docker volume create "$(volume_name "$vol")" >/dev/null
    # Clear first: restoring over a populated volume would merge two
    # different points in time.
    docker run --rm \
      -v "$(volume_name "$vol"):/dst" \
      -v "$(cd "$dir" && pwd):/src:ro" \
      alpine:3.20 sh -c "rm -rf /dst/* /dst/..?* 2>/dev/null; tar xzf /src/${vol}.tar.gz -C /dst"
    log "restored $vol"
  done

  log "starting services"
  docker compose -f "$COMPOSE_FILE" up -d --wait
  log "restored document_count=$(document_count || echo '?')"
}

case "${1:-}" in
  backup)  shift; do_backup "$@" ;;
  verify)  shift; do_verify "$@" ;;
  restore) shift; do_restore "$@" ;;
  *) cat >&2 <<EOF
usage: $0 {backup|verify|restore} [dir]

  backup  [dir]   archive chroma_data, checkpoint_data and minio_data
  verify  <dir>   prove the archives are readable and non-empty
  restore <dir>   verify, then stop services, replace volumes, restart
EOF
     exit 2 ;;
esac
