from __future__ import annotations

import json
import fnmatch
from pathlib import Path
from typing import Iterable

from sweagent.environment.swe_env import SWEEnv
from sweagent.run.hooks.abstract import RunHook
from sweagent.types import AgentRunResult
from sweagent.utils.log import get_logger


class CopyContainerArtifactsHook(RunHook):
    """
    Copies agent-generated files from the Docker environment to the local trajectory dir.

    Strategy:
    - Detect working_dir inside the container (via /root/state.json if available, else /<repo_name>).
    - Use `git ls-files -mo --exclude-standard` in working_dir to list modified/untracked files.
    - Filter by include/exclude patterns.
    - Read files via env.read_file and write to output_dir/<instance_id>/<output_subdir>/...

    Notes:
    - Only text files are copied (read_file returns text). Adjust if you need binaries.
    - To constrain size, add a max_bytes guard or refine include_patterns.
    """

    def __init__(
        self,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        output_subdir: str = "artifacts",
        use_git_detection: bool = True,
    ) -> None:
        self.logger = get_logger("swea-copy-artifacts", emoji="📥")
        self._include_patterns = include_patterns or [
            # sensible defaults for codegen flows
            "*.py",
            "*.json",
            "*.txt",
            "*.md",
            "*.log",
            "*.yaml",
            "*.yml",
        ]
        self._exclude_patterns = exclude_patterns or [
            # skip common heavy/noisy files
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            ".venv/*",
            "venv/*",
            "node_modules/*",
        ]
        self._output_subdir = output_subdir
        self._use_git_detection = use_git_detection

    # Run context captured during lifecycle
    def on_init(self, *, run) -> None:
        self._run = run
        self._output_dir = Path(run.output_dir)

    def on_instance_start(self, *, index: int, env: SWEEnv, problem_statement) -> None:
        self._env = env
        self._problem_statement = problem_statement

    def on_instance_completed(self, *, result: AgentRunResult):
        try:
            instance_id = self._problem_statement.id
            workdir = self._detect_working_dir()
            if not workdir:
                self.logger.warning("No working_dir detected; skipping artifact copy")
                return

            candidates = self._list_candidates(workdir)
            selected = self._apply_filters(candidates, base=workdir)

            if not selected:
                self.logger.info("No artifacts to copy (after filtering)")
                return

            dest_root = Path(self._output_dir) / instance_id / self._output_subdir
            for abs_path in selected:
                rel = abs_path.removeprefix(workdir).lstrip("/")  # relative to workdir
                target = dest_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    content = self._env.read_file(abs_path, encoding="utf-8", errors="replace")
                except Exception as e:
                    self.logger.warning(f"Skipping unreadable file {abs_path}: {e}")
                    continue
                target.write_text(content, encoding="utf-8")
            self.logger.info(f"Copied {len(selected)} artifact(s) to {dest_root}")
        except Exception as e:
            # Never fail the run because of artifact copying
            self.logger.error(f"Artifact copy failed: {e}")

    # --- helpers ---

    def _detect_working_dir(self) -> str | None:
        # Preferred: /root/state.json may include {"working_dir": "/<repo_name or path>"}
        try:
            state = self._env.read_file("/root/state.json")
            data = json.loads(state)
            wd = data.get("working_dir")
            if isinstance(wd, str) and wd.strip():
                return wd
        except Exception:
            pass

        # Fallback: infer from repo config if available
        try:
            repo = getattr(self._env, "repo", None)
            repo_name = getattr(repo, "repo_name", None)
            if isinstance(repo_name, str) and repo_name.strip():
                return f"/{repo_name}"
        except Exception:
            pass

        return None

    def _list_candidates(self, workdir: str) -> list[str]:
        files: list[str] = []

        if self._use_git_detection:
            try:
                output = self._env.communicate(
                    f'cd "{workdir}" && git ls-files -mo --exclude-standard',
                    check="warn",
                    timeout=20,
                )
                for line in output.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    files.append(f"{workdir.rstrip('/')}/{line}")
            except Exception:
                # Ignore and fall back to a small known set below
                pass

        # Always include some common artifact filenames in the repo root
        common = ["concise.py", "touched_files.txt", "traced_test.json", "pruned_test_file.py"]
        for name in common:
            abs_path = f"{workdir.rstrip('/')}/{name}"
            try:
                # cheap existence check
                out = self._env.communicate(
                    f'test -f "{abs_path}" && echo EXISTS || echo MISSING',
                    check="ignore",
                    timeout=5,
                )
                if 'EXISTS' in out:
                    # If the command ran, we still need an explicit content check before copying
                    files.append(abs_path)
            except Exception:
                pass

        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for p in files:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result

    def _apply_filters(self, files: Iterable[str], base: str) -> list[str]:
        def matches_any(path: str, patterns: list[str]) -> bool:
            rel = path.removeprefix(base).lstrip("/")
            return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path, pat) for pat in patterns)

        selected: list[str] = []
        for f in files:
            if self._exclude_patterns and matches_any(f, self._exclude_patterns):
                continue
            if self._include_patterns and not matches_any(f, self._include_patterns):
                continue
            selected.append(f)
        return selected