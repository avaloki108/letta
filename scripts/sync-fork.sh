#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
FORK_URL="${FORK_URL:-https://github.com/avaloki108/letta.git}"
LOCAL_BRANCH="${LOCAL_BRANCH:-main}"
FORK_BRANCH="${FORK_BRANCH:-main}"
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
echo "Pushing $LOCAL_BRANCH -> $FORK_URL:$FORK_BRANCH"
git push origin "HEAD:$FORK_BRANCH"
local_sha="$(git rev-parse HEAD)"
remote_sha="$(git ls-remote "$FORK_URL" "refs/heads/$FORK_BRANCH" | awk '{print $1}')"
echo "Local HEAD : $local_sha"
echo "Fork branch: $remote_sha"
[[ -n "$remote_sha" && "$local_sha" == "$remote_sha" ]]
echo "Fork sync complete."
