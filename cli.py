import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Allow `python cli.py ...` without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voice_ai_banking_support_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

