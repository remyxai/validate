"""Git plumbing: a cached mirror per repo, immutable ref resolution, and
per-arm checkouts that are full clones (so they work inside a container)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request

from .canonical import digest

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PR_URL_RE = re.compile(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/pull/(\d+)")


class GitError(Exception):
    pass


def git(args, cwd=None, check=True) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def is_remote(repo: str) -> bool:
    return "://" in repo or repo.startswith("git@")


def resolve_repo(repo: str, project: str) -> str:
    if is_remote(repo):
        return repo
    path = repo if os.path.isabs(repo) else os.path.normpath(os.path.join(project, repo))
    if not os.path.isdir(path):
        raise GitError(f"repo path does not exist: {path}")
    return path


def repo_identity(repo: str, project: str) -> str:
    """A machine-independent name for the repo, used in change_id."""
    location = resolve_repo(repo, project)
    url = location
    if not is_remote(location):
        url = git(["config", "--get", "remote.origin.url"], cwd=location, check=False)
        if not url:
            return "local:" + os.path.basename(os.path.abspath(location))
    url = re.sub(r"^git@([^:]+):", r"https://\1/", url.strip())
    url = re.sub(r"\.git$", "", url).rstrip("/")
    return url.lower()


def mirror_path(repo: str, project: str) -> str:
    location = resolve_repo(repo, project)
    return os.path.join(project, ".validate", "cache", "repos", digest(location)[:16] + ".git")


def ensure_mirror(repo: str, project: str) -> str:
    location = resolve_repo(repo, project)
    path = mirror_path(repo, project)
    if not os.path.isdir(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        git(["clone", "--quiet", "--bare", location, path])
    return path


def _has_commit(mirror: str, sha: str) -> bool:
    return bool(git(["rev-parse", "--verify", "--quiet", sha + "^{commit}"], cwd=mirror, check=False))


def _keep(mirror: str, sha: str) -> None:
    git(["update-ref", f"refs/validate/keep/{sha}", sha], cwd=mirror)


def resolve_ref(mirror: str, ref: str) -> str:
    """Resolve a branch, tag, PR ref or sha to a commit sha, fetching if needed.

    A fetch failure is tolerated when the ref already resolves from the mirror,
    so a repeat run works offline.
    """
    if SHA_RE.match(ref):
        if not _has_commit(mirror, ref):
            git(["fetch", "--quiet", "origin", ref], cwd=mirror, check=False)
        if not _has_commit(mirror, ref):
            raise GitError(f"commit {ref} not found in the repo")
        _keep(mirror, ref)
        return ref
    candidates = [ref]
    m = re.match(r"^(?:refs/)?pull/(\d+)(?:/head)?$", ref)
    if m:
        candidates = [f"refs/pull/{m.group(1)}/head"]
    sha = ""
    for cand in candidates:
        proc = subprocess.run(["git", "fetch", "--quiet", "origin", cand], cwd=mirror,
                              capture_output=True, text=True,
                              env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        if proc.returncode == 0:
            sha = git(["rev-parse", "FETCH_HEAD^{commit}"], cwd=mirror, check=False)
            if sha:
                break
    if not sha:
        for cand in (ref, f"refs/heads/{ref}", f"refs/tags/{ref}"):
            sha = git(["rev-parse", "--verify", "--quiet", cand + "^{commit}"], cwd=mirror, check=False)
            if sha:
                break
    if not sha:
        raise GitError(f"cannot resolve ref {ref!r}")
    _keep(mirror, sha)
    return sha


def checkout(mirror: str, sha: str, dest: str) -> None:
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    git(["init", "--quiet"], cwd=dest)
    git(["fetch", "--quiet", mirror, f"refs/validate/keep/{sha}"], cwd=dest)
    git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=dest)
    head = git(["rev-parse", "HEAD"], cwd=dest)
    if head != sha:
        raise GitError(f"checkout produced {head}, expected {sha}")


def tracked_changes(repo_dir: str) -> list:
    """Tracked files whose contents differ from HEAD. Untracked build products
    (a virtualenv, egg-info) are not source and are ignored."""
    out = git(["diff", "--name-only", "HEAD"], cwd=repo_dir, check=False)
    return [line for line in out.splitlines() if line.strip()]


def head_sha(repo_dir: str) -> str:
    return git(["rev-parse", "HEAD"], cwd=repo_dir, check=False)


def merge_base(mirror: str, a: str, b: str) -> str:
    return git(["merge-base", a, b], cwd=mirror, check=False)


def parse_pr_url(url: str):
    m = PR_URL_RE.search(url)
    if not m:
        raise GitError(f"not a GitHub pull request URL: {url}")
    return m.group(1), m.group(2), int(m.group(3))


def pr_info(url: str) -> dict:
    """Base and head of a GitHub PR. Talks to api.github.com and nothing else."""
    owner, name, number = parse_pr_url(url)
    api = f"repos/{owner}/{name}/pulls/{number}"
    data = None
    if shutil.which("gh"):
        proc = subprocess.run(["gh", "api", api], capture_output=True, text=True)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
    if data is None:
        req = urllib.request.Request(f"https://api.github.com/{api}",
                                     headers={"Accept": "application/vnd.github+json",
                                              "User-Agent": "remyx-validate"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    return {
        "repo": f"https://github.com/{owner}/{name}",
        "number": number,
        "title": data.get("title", ""),
        "base_sha": data["base"]["sha"],
        "base_ref": data["base"]["ref"],
        "head_sha": data["head"]["sha"],
        "draft": bool(data.get("draft")),
        "url": f"https://github.com/{owner}/{name}/pull/{number}",
    }
