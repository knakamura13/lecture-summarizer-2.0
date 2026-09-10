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
