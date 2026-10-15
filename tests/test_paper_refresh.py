import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import paper_refresh


class PaperPublicationTests(unittest.TestCase):
    def test_missing_checkpoints_never_publish_initialized_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(paper_refresh, "ROOT", root), \
                 patch.object(paper_refresh, "checkpoint_signature", return_value={}), \
                 patch.object(paper_refresh.subprocess, "run") as run:
                paper_refresh.main()
            run.assert_not_called()
            state = json.loads((root / "outputs/paper_exports/refresh_status.json").read_text())
            self.assertEqual(state["status"], "waiting_checkpoints")

    def test_retry_publishes_only_artifacts_and_preserves_private_commits_and_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, source = root / "remote.git", root / "source"

            def git(*args, cwd=source):
                return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()

            subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
            source.mkdir()
            git("init", "-b", "main")
            git("config", "user.name", "Publication Test")
            git("config", "user.email", "publication@example.invalid")
            git("config", "commit.gpgsign", "false")
            (source / ".gitignore").write_text("outputs/\n")
            (source / "paper").mkdir()
            (source / "benchmarks").mkdir()
            (source / "paper/report.md").write_text("old report\n")
            (source / "benchmarks/raw.json").write_text("{}\n")
            (source / "private.txt").write_text("public version\n")
            (source / "benchmark_cpu.py").write_text("from pathlib import Path\nPath('benchmarks/raw.json').write_text('{\"measured\":true}\\n')\n")
            (source / "paper_results.py").write_text("from pathlib import Path\nPath('paper/report.md').write_text('measured report\\n')\np=Path('outputs/render_count');p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n")
            git("add", ".")
            git("commit", "-m", "public baseline")
            git("remote", "add", "origin", str(remote))
            git("push", "-u", "origin", "main")
            (source / "private.txt").write_text("unpublished private work\n")
            git("add", "private.txt")
            git("commit", "-m", "private unpublished commit")
            private_head = git("rev-parse", "HEAD")
            (source / "private_staged.txt").write_text("keep staged\n")
            git("add", "private_staged.txt")
            (source / "private_untracked.txt").write_text("keep untracked\n")
            real_run = subprocess.run

            def fail_push(command, *args, **kwargs):
                if command[:2] == ["git", "push"]:
                    raise subprocess.CalledProcessError(1, command)
                return real_run(command, *args, **kwargs)

            with patch.object(paper_refresh, "ROOT", source), \
                 patch.object(paper_refresh, "ARTIFACTS", ["paper/report.md", "benchmarks/raw.json"]), \
                 patch.object(paper_refresh, "checkpoint_signature", return_value={"model": "sha256"}), \
                 patch.object(paper_refresh, "github_environment", return_value=dict(os.environ)):
                with patch.object(paper_refresh.subprocess, "run", side_effect=fail_push):
                    with self.assertRaises(subprocess.CalledProcessError):
                        paper_refresh.main()
                path = source / "outputs/paper_exports/refresh_status.json"
                pending = json.loads(path.read_text())
                self.assertEqual(pending["status"], "pending")
                paper_refresh.main()
                final = json.loads(path.read_text())
                self.assertEqual(final["status"], "published")
                self.assertEqual(final["commit"], pending["generated_commit"])
                self.assertEqual((source / "outputs/render_count").read_text(), "1")
                paper_refresh.main()  # No completed checkpoint changed: no new commit/render.
                self.assertEqual((source / "outputs/render_count").read_text(), "1")
            self.assertEqual(git("rev-parse", "HEAD"), private_head)
            self.assertEqual(git("diff", "--cached", "--name-only"), "private_staged.txt")
            self.assertTrue((source / "private_untracked.txt").exists())
            self.assertEqual(git("show", "main:private.txt", cwd=remote), "public version")
            self.assertEqual(git("show", "main:paper/report.md", cwd=remote), "measured report")
            self.assertNotIn("private_staged.txt", git("ls-tree", "-r", "--name-only", "main", cwd=remote))


if __name__ == "__main__":
    unittest.main()
