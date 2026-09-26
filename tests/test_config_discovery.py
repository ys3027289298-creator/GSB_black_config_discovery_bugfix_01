"""Tests for deterministic configuration discovery and source selection.

Black's mapping from input paths to formatting mode must be a deterministic
computation: running from the project root, from a subdirectory, with
explicit files, directories, stdin, or ``--stdin-filename`` must all yield
the same configuration and the same file set for the same project.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as imp_version
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from click.testing import CliRunner
from packaging.version import Version

import black
import black.files
from black.report import Report
from tests.util import change_directory


class BlackRunner(CliRunner):
    """Keep STDOUT and STDERR separate, mirroring tests.test_black."""

    def __init__(self) -> None:
        if Version(imp_version("click")) >= Version("8.2.0"):
            super().__init__()
        else:
            super().__init__(mix_stderr=False)  # type: ignore


# Formats to a single line at line-length 88, but is split at line-length 40.
SOURCE = 'foo("aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc")\n'
SOURCE_40 = 'foo(\n    "aaaaaaaaaa",\n    "bbbbbbbbbb",\n    "cccccccccc",\n)\n'
CONFIG_40 = "[tool.black]\nline-length = 40\n"
CONFIG_100 = "[tool.black]\nline-length = 100\n"


def make_project(root: Path, config: str = CONFIG_40) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(config, encoding="utf-8")
    (root / ".git").mkdir(exist_ok=True)
    src = root / "src"
    src.mkdir(exist_ok=True)
    (src / "a.py").write_text(SOURCE, encoding="utf-8")
    return src


def invoke(args: list[str], cwd: Path, input: str | None = None):  # noqa: A002
    runner = BlackRunner()
    with change_directory(cwd):
        return runner.invoke(black.main, args, input=input)


class TestDeterministicEntryPoints:
    """Every entry point must find the same config and format identically."""

    def test_run_from_project_root_with_directory(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            result = invoke(["."], cwd=root)
            assert result.exit_code == 0, result.output
            assert (src / "a.py").read_text() == SOURCE_40

    def test_run_from_subdirectory(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            result = invoke(["."], cwd=src)
            assert result.exit_code == 0, result.output
            assert (src / "a.py").read_text() == SOURCE_40

    def test_explicit_file_from_other_cwd(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            other = root / "elsewhere"
            other.mkdir()
            result = invoke([str(src / "a.py")], cwd=other)
            assert result.exit_code == 0, result.output
            assert (src / "a.py").read_text() == SOURCE_40

    def test_explicit_directory_from_other_cwd(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            other = root / "elsewhere"
            other.mkdir()
            result = invoke([str(src)], cwd=other)
            assert result.exit_code == 0, result.output
            assert (src / "a.py").read_text() == SOURCE_40

    def test_stdin_uses_cwd_project_config(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            make_project(root)
            result = invoke(["-"], cwd=root, input=SOURCE)
            assert result.exit_code == 0, result.output
            assert result.stdout == SOURCE_40

    def test_stdin_filename_locates_config(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            make_project(root)
            other = root / "elsewhere"
            other.mkdir()
            result = invoke(
                ["-", "--stdin-filename", str(root / "src" / "a.py")],
                cwd=other,
                input=SOURCE,
            )
            assert result.exit_code == 0, result.output
            assert result.stdout == SOURCE_40

    def test_stdin_is_not_scanned_as_directory(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            result = invoke(["-"], cwd=root, input=SOURCE)
            assert result.exit_code == 0, result.output
            # The file on disk must remain untouched: stdin is not a scan root.
            assert (src / "a.py").read_text() == SOURCE

    def test_stdin_filename_force_excluded(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            make_project(root, CONFIG_40 + "force-exclude = 'a\\.py'\n")
            result = invoke(
                ["-", "--stdin-filename", "src/a.py"], cwd=root, input=SOURCE
            )
            assert result.exit_code == 0, result.output
            # Excluded stdin is passed through unchanged.
            assert result.stdout == SOURCE

    def test_command_line_overrides_config(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            result = invoke(["--line-length", "88", "src/a.py"], cwd=root)
            assert result.exit_code == 0, result.output
            # 88 columns wins over the configured 40: nothing to reformat.
            assert (src / "a.py").read_text() == SOURCE

    def test_nested_project_uses_nearest_config(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            outer_src = make_project(root, CONFIG_100)
            inner = root / "inner"
            inner_src = make_project(inner, CONFIG_40)
            # Running inside the nested project picks up its own config.
            result = invoke(["."], cwd=inner)
            assert result.exit_code == 0, result.output
            assert (inner_src / "a.py").read_text() == SOURCE_40
            # The outer project is unaffected by the nested config.
            result = invoke(["src/a.py"], cwd=root)
            assert result.exit_code == 0, result.output
            assert (outer_src / "a.py").read_text() == SOURCE

    def test_no_config_falls_back_to_default(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            (root / ".git").mkdir()
            src = root / "a.py"
            src.write_text(SOURCE, encoding="utf-8")
            result = invoke(["a.py"], cwd=root)
            assert result.exit_code == 0, result.output
            # Default line length is 88: the source is left alone.
            assert src.read_text() == SOURCE


class TestInvalidConfiguration:
    def test_invalid_toml_exits_via_error_path(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            root.joinpath("pyproject.toml").write_text(
                "this is not [valid toml\n", encoding="utf-8"
            )
            root.joinpath("a.py").write_text(SOURCE, encoding="utf-8")
            result = invoke(["a.py"], cwd=root)
            assert result.exit_code == 1
            assert "Error reading configuration file" in result.stderr
            assert "Traceback" not in result.stderr

    def test_invalid_toml_never_splices_stale_cache(self) -> None:
        with TemporaryDirectory() as ws:
            config = Path(ws) / "pyproject.toml"
            config.write_text(CONFIG_40, encoding="utf-8")
            assert black.parse_pyproject_toml(str(config))["line_length"] == 40
            # Rewrite the file invalid: the stale cached parse must not leak.
            config.write_text("not [valid\n", encoding="utf-8")
            with pytest.raises(black.files.tomllib.TOMLDecodeError):
                black.parse_pyproject_toml(str(config))
            # Root discovery still deterministically finds the project.
            root, method = black.find_project_root((str(Path(ws) / "a.py"),))
            assert root == Path(ws).resolve()
            assert method == "pyproject.toml"
            # A valid rewrite is picked up fresh, not merged with old state.
            config.write_text(CONFIG_100, encoding="utf-8")
            assert black.parse_pyproject_toml(str(config))["line_length"] == 100


class TestDuplicateAndSymlinkedPaths:
    def test_duplicate_paths_format_once(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            result = invoke(
                ["--check", "src/a.py", "./src/a.py", "src/../src/a.py"], cwd=root
            )
            assert result.exit_code == 1
            assert "1 file would be reformatted" in result.stderr

    def test_get_sources_deduplicates(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws).resolve()
            src = make_project(root)
            report = Report()
            with change_directory(root):
                sources = black.get_sources(
                    root=root,
                    src=(
                        "src/a.py",
                        str(src / "a.py"),
                        "src/../src/a.py",
                        "src",
                    ),
                    quiet=False,
                    verbose=False,
                    include=black.re_compile_maybe_verbose(black.DEFAULT_INCLUDES),
                    exclude=None,
                    extend_exclude=None,
                    force_exclude=None,
                    report=report,
                    stdin_filename=None,
                )
            assert len(sources) == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics")
    def test_symlinked_path_not_formatted_twice(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            link = root / "link"
            link.symlink_to(src)
            result = invoke(
                ["--check", "link/a.py", "src/a.py"], cwd=root
            )
            assert result.exit_code == 1
            assert "1 file would be reformatted" in result.stderr

    @pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics")
    def test_symlink_outside_root_ignored_and_force_exclude_honored(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws) / "proj"
            src = make_project(root, CONFIG_40 + 'force-exclude = "skipme"\n')
            outside = Path(ws) / "outside"
            outside.mkdir()
            (outside / "b.py").write_text(SOURCE, encoding="utf-8")
            (src / "link.py").symlink_to(outside / "b.py")
            (src / "skipme.py").write_text(SOURCE, encoding="utf-8")
            result = invoke(["src"], cwd=root)
            assert result.exit_code == 0, result.output
            # The real file is formatted...
            assert (src / "a.py").read_text() == SOURCE_40
            # ...the force-excluded file is untouched...
            assert (src / "skipme.py").read_text() == SOURCE
            # ...and the symlink escaping the root is not followed.
            assert (outside / "b.py").read_text() == SOURCE


class TestModeConsistency:
    def test_check_does_not_write(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            result = invoke(["--check", "src/a.py"], cwd=root)
            assert result.exit_code == 1
            assert (src / "a.py").read_text() == SOURCE

    def test_diff_does_not_write_and_matches_format(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            diff_result = invoke(["--diff", "src/a.py"], cwd=root)
            assert diff_result.exit_code == 0, diff_result.output
            # Diff mode never writes...
            assert (src / "a.py").read_text() == SOURCE
            # ...and it previews exactly what formatting produces.
            assert '+    "aaaaaaaaaa",' in diff_result.output
            format_result = invoke(["src/a.py"], cwd=root)
            assert format_result.exit_code == 0, format_result.output
            assert (src / "a.py").read_text() == SOURCE_40
            # A subsequent check in the same mode now passes.
            check_result = invoke(["--check", "src/a.py"], cwd=root)
            assert check_result.exit_code == 0


class TestStateIsolation:
    def discover(self, path: Path) -> tuple[Path | None, str, int]:
        srcs = (str(path),)
        root, _ = black.find_project_root(srcs)
        config_path = black.files.find_pyproject_toml(srcs)
        assert config_path is not None
        config = black.parse_pyproject_toml(config_path)
        return root, config_path, config["line_length"]

    def test_concurrent_projects_do_not_cross_wire(self) -> None:
        with TemporaryDirectory() as ws:
            proj_a = Path(ws) / "proj_a"
            proj_b = Path(ws) / "proj_b"
            src_a = make_project(proj_a, CONFIG_40)
            src_b = make_project(proj_b, CONFIG_100)
            cwd_before = os.getcwd()
            errors: list[BaseException] = []
            lock = threading.Lock()

            def work(file: Path, expected_root: Path, expected_ll: int) -> None:
                try:
                    for _ in range(10):
                        root, config_path, line_length = self.discover(file)
                        assert root == expected_root.resolve()
                        assert config_path == str(
                            expected_root.resolve() / "pyproject.toml"
                        )
                        assert line_length == expected_ll
                        assert os.getcwd() == cwd_before
                except BaseException as exc:  # noqa: BLE001
                    with lock:
                        errors.append(exc)

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(work, src_a / "a.py", proj_a, 40),
                    pool.submit(work, src_b / "a.py", proj_b, 100),
                ]
                for future in futures:
                    future.result()
            assert not errors
            assert os.getcwd() == cwd_before

    def test_concurrent_formatting_uses_own_mode_and_report(self) -> None:
        with TemporaryDirectory() as ws:
            proj_a = Path(ws) / "proj_a"
            proj_b = Path(ws) / "proj_b"
            src_a = make_project(proj_a, CONFIG_40)
            src_b = make_project(proj_b, CONFIG_100)
            mode_a = black.Mode(line_length=40)
            mode_b = black.Mode(line_length=100)
            report_a = Report(check=False, diff=False, quiet=True)
            report_b = Report(check=False, diff=False, quiet=True)

            def fmt(path: Path, mode: black.Mode, report: Report) -> None:
                changed = black.format_file_in_place(
                    path, fast=True, mode=mode, write_back=black.WriteBack.YES
                )
                report.done(path, black.Changed.YES if changed else black.Changed.NO)

            with ThreadPoolExecutor(max_workers=4) as pool:
                fa = pool.submit(fmt, src_a / "a.py", mode_a, report_a)
                fb = pool.submit(fmt, src_b / "a.py", mode_b, report_b)
                fa.result()
                fb.result()

            assert (src_a / "a.py").read_text() == SOURCE_40
            assert (src_b / "a.py").read_text() == SOURCE
            assert report_a.change_count == 1
            assert report_b.same_count == 1
            assert report_a.failure_count == 0
            assert report_b.failure_count == 0

    def test_repeated_api_calls_are_deterministic(self) -> None:
        with TemporaryDirectory() as ws:
            root = Path(ws)
            src = make_project(root)
            first = self.discover(src / "a.py")
            for _ in range(3):
                assert self.discover(src / "a.py") == first
            # Editing the config is reflected on the next call.
            (root / "pyproject.toml").write_text(CONFIG_100, encoding="utf-8")
            assert self.discover(src / "a.py")[2] == 100
            # And discovery remains stable afterwards.
            assert self.discover(src / "a.py") == self.discover(src / "a.py")
