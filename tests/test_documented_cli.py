from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_help_lists_every_documented_flag() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    for flag in (
        "--input",
        "--output",
        "--model",
        "--provider",
        "--ollama-host",
        "--timeout",
        "--max-retries",
        "--target-words",
        "--chunk-tokens",
        "--overlap-tokens",
        "--max-merge-children",
        "--verify",
        "--max-repair-passes",
        "--citations",
        "--audit",
        "--cache-dir",
        "--run-id",
        "--resume",
        "--max-concurrency",
        "--dry-run",
        "--strategy",
        "--context-window",
        "--max-output-tokens",
        "--safety-margin-tokens",
        "--safety-margin-fraction",
        "--max-direct-tokens",
    ):
        assert flag in result.stdout
    for description in (
        "how to execute",
        "the model's total context size",
        "tokens reserved for the response when sizing a request",
        "minimum tokens held back from the context window",
        "fraction of the context window held back",
        "force hierarchical",
    ):
        assert description in result.stdout


def test_entrypoint_applies_strategy_budget_flags_during_dry_run(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("A source sentence.", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--input",
            str(source),
            "--strategy",
            "hierarchical",
            "--context-window",
            "10000",
            "--max-output-tokens",
            "32",
            "--safety-margin-tokens",
            "8",
            "--safety-margin-fraction",
            "0",
            "--max-direct-tokens",
            "100",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Strategy: hierarchical" in result.stdout
    assert "Context window: 10000 tokens" in result.stdout


def test_readme_documents_current_cli_and_credentials() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY" in readme
    assert "--provider ollama" in readme
    assert "--ollama-host" in readme
    assert "qwen3.8" in readme
    assert "gemma3:4b" in readme
    assert "ollama serve" in readme
    assert "does not require an API key" in readme
    assert "python main.py --input source.txt --output summary.txt" in readme
    assert "python main.py --dry-run" in readme
    assert "gpt-4o-mini" in readme
    assert "nonzero" in readme
