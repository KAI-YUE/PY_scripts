#!/usr/bin/env bash
set -Eeuo pipefail

# clean_git_stable_push.sh
#
# Purpose:
#   Replace the current Git history with one new "stable version" commit,
#   while first saving the old .git folder and online refs in an out_src backup.
#
# Usage:
#   chmod +x clean_git_stable_push.sh restore_git_backup.sh
#   ./clean_git_stable_push.sh
#
# Optional environment variables:
#   BRANCH=main ./clean_git_stable_push.sh
#   REMOTE_NAME=origin ./clean_git_stable_push.sh
#   OUT_SRC_DIR=/absolute/path/to/out_src ./clean_git_stable_push.sh
#   DRY_RUN=1 ./clean_git_stable_push.sh
#   SKIP_PUSH=1 ./clean_git_stable_push.sh
#   PUSH_MODE=force ./clean_git_stable_push.sh
#
# Notes:
#   - This rewrites the remote branch history.
#   - Default push mode uses --force-with-lease against the remote commit
#     detected before deleting .git. If no remote branch is found, it falls
#     back to --force.

timestamp() {
	date +"%Y%m%d_%H%M%S"
}

need_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "ERROR: Missing required command: $1" >&2
		exit 1
	fi
}

real_path() {
	if command -v realpath >/dev/null 2>&1; then
		realpath "$1"
	else
		python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
	fi
}

need_cmd git
need_cmd tar

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".git" ]]; then
	echo "ERROR: No .git directory found in: $REPO_ROOT" >&2
	echo "Run this from inside the repository you want to clean." >&2
	exit 1
fi

REMOTE_NAME="${REMOTE_NAME:-origin}"
REMOTE_URL="$(git remote get-url "$REMOTE_NAME" 2>/dev/null || true)"

if [[ -z "$REMOTE_URL" ]]; then
	echo "ERROR: Could not find remote '$REMOTE_NAME'." >&2
	echo "Add one first, or run with REMOTE_NAME=your_remote_name." >&2
	exit 1
fi

BRANCH="${BRANCH:-$(git branch --show-current 2>/dev/null || true)}"
if [[ -z "$BRANCH" ]]; then
	BRANCH="${DEFAULT_BRANCH:-main}"
fi

REPO_NAME="$(basename "$REPO_ROOT")"
OUT_SRC_DIR="${OUT_SRC_DIR:-"$(dirname "$REPO_ROOT")/out_src"}"
BACKUP_DIR="$OUT_SRC_DIR/git_clean_backup_${REPO_NAME}_$(timestamp)"

mkdir -p "$BACKUP_DIR"

echo "Repository: $REPO_ROOT"
echo "Backup dir: $BACKUP_DIR"
echo "Remote: $REMOTE_NAME -> $REMOTE_URL"
echo "Branch: $BRANCH"
echo

echo "Saving local Git metadata and online references..."
git status --short > "$BACKUP_DIR/status_before.txt" || true
git log -1 --format=fuller > "$BACKUP_DIR/latest_commit_before.txt" || true
git branch -vv > "$BACKUP_DIR/branches_before.txt" || true
git remote -v > "$BACKUP_DIR/remotes_before.txt" || true
git show-ref > "$BACKUP_DIR/show_ref_before.txt" || true
git ls-files -co --exclude-standard > "$BACKUP_DIR/worktree_files_before.txt" || true

printf "%s\n" "$REPO_ROOT" > "$BACKUP_DIR/repo_root.txt"
printf "%s\n" "$REMOTE_NAME" > "$BACKUP_DIR/remote_name.txt"
printf "%s\n" "$REMOTE_URL" > "$BACKUP_DIR/remote_url.txt"
printf "%s\n" "$BRANCH" > "$BACKUP_DIR/branch.txt"

git ls-remote --refs "$REMOTE_URL" > "$BACKUP_DIR/online_refs_before.txt" || true
REMOTE_EXPECT="$(git ls-remote --heads "$REMOTE_URL" "$BRANCH" | awk '{print $1}' || true)"
printf "%s\n" "$REMOTE_EXPECT" > "$BACKUP_DIR/remote_expected_commit.txt"

echo "Compressing old .git directory..."
tar -C "$REPO_ROOT" -czf "$BACKUP_DIR/dot_git_backup.tar.gz" .git

cat > "$BACKUP_DIR/README.txt" <<README
This folder was created by clean_git_stable_push.sh.

It contains:
- dot_git_backup.tar.gz: compressed backup of the old .git directory
- online_refs_before.txt: remote refs before the rewrite
- remote_expected_commit.txt: expected remote commit for the rewritten branch
- remote_url.txt: saved online Git remote URL
- branch.txt: branch that was rewritten
- status_before.txt / branches_before.txt / show_ref_before.txt: local Git state before cleanup

To restore old local Git metadata:
	./restore_git_backup.sh "$BACKUP_DIR"

Important:
	Restoring local .git does not automatically undo the remote force-push.
	To put the old history back online, restore .git first, then push the old branch manually.
README

if [[ "${DRY_RUN:-0}" == "1" ]]; then
	echo
	echo "DRY_RUN=1 enabled. Backup created, but .git was not removed."
	exit 0
fi

echo
echo "Removing old .git directory..."
rm -rf "$REPO_ROOT/.git"

echo "Initializing fresh Git repository..."
git init
git checkout -b "$BRANCH" >/dev/null 2>&1 || git branch -M "$BRANCH"
git remote add "$REMOTE_NAME" "$REMOTE_URL"

# If the backup folder is inside the repo, keep it out of the new commit.
REPO_ABS="$(real_path "$REPO_ROOT")"
OUT_ABS="$(real_path "$OUT_SRC_DIR")"
if [[ "$OUT_ABS" == "$REPO_ABS/"* ]]; then
	OUT_REL="${OUT_ABS#"$REPO_ABS"/}"
	printf "\n# local backup folder created by clean_git_stable_push.sh\n/%s/\n" "$OUT_REL" >> "$REPO_ROOT/.git/info/exclude"
fi

echo "Creating one new stable commit from the current working tree..."
git add -A

if git diff --cached --quiet; then
	echo "ERROR: Nothing was staged for commit." >&2
	echo "Your .git backup is still safe here: $BACKUP_DIR" >&2
	exit 1
fi

COMMIT_MSG="stable ver. $(date '+%Y-%m-%d %H:%M:%S %z')"
git commit -m "$COMMIT_MSG"

echo
echo "New commit created:"
git --no-pager log --oneline -1
echo

if [[ "${SKIP_PUSH:-0}" == "1" ]]; then
	echo "SKIP_PUSH=1 enabled. Not pushing."
	echo "Backup kept at: $BACKUP_DIR"
	exit 0
fi

echo "Pushing new stable history to remote..."
if [[ -n "$REMOTE_EXPECT" && "${PUSH_MODE:-lease}" == "lease" ]]; then
	git push --force-with-lease="refs/heads/$BRANCH:$REMOTE_EXPECT" -u "$REMOTE_NAME" "HEAD:refs/heads/$BRANCH"
else
	git push --force -u "$REMOTE_NAME" "HEAD:refs/heads/$BRANCH"
fi

echo
echo "Done."
echo "Old .git backup kept at:"
echo "  $BACKUP_DIR"
echo
echo "Do not delete the backup until you verify the remote repo looks correct."
