# Fork sync workflow

This repo uses a main-only custom workflow:
- local custom branch: `main`
- fork remote: `origin -> https://github.com/avaloki108/letta.git`
- upstream remote: `upstream -> https://github.com/letta-ai/letta.git`

## Common commands

### Update local main from upstream
```bash
./scripts/update-from-upstream.sh
```

### Sync local main to your fork
```bash
./scripts/sync-fork.sh
```

### Update from upstream and sync fork in one shot
```bash
./scripts/update-and-sync.sh
```

## Safety
- scripts expect you to be on `main`
- scripts refuse to run with a dirty worktree
- they push only to `origin` (your fork), never `upstream`

To override the dirty-tree check:
```bash
FORCE_DIRTY=1 ./scripts/sync-fork.sh
```

## Conflict recovery
```bash
git status
# fix files
git add <files>
git rebase --continue
```
Abort a rebase with:
```bash
git rebase --abort
```
