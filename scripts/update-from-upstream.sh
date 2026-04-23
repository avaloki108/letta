#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
LOCAL_BRANCH="${LOCAL_BRANCH:-main}"
FORCE_DIRTY="${FORCE_DIRTY:-0}"
current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$LOCAL_BRANCH" ]]; then
  echo "error: expected branch '$LOCAL_BRANCH' but found '$current_branch'" >&2
  exit 1
fi
if [[ "$FORCE_DIRTY" != "1" ]] && [[ -n "$(git status --short)" ]]; then
  echo "error: working tree is dirty. Commit/stash first, or rerun with FORCE_DIRTY=1." >&2
  git status --short
  exit 1
fi
echo "Fetching upstream/$UPSTREAM_BRANCH"
git fetch upstream "$UPSTREAM_BRANCH"
echo "Rebasing $LOCAL_BRANCH onto upstream/$UPSTREAM_BRANCH"
git rebase "upstream/$UPSTREAM_BRANCH"
echo "Upstream update complete."
