"""Thin wrapper so you can run the CLI from a checkout without installing the package.

    python run.py --source "Paris is the capital of France." --answer "France's capital is Paris."

The actual logic lives in ``groundcheck/cli.py`` (also installed as the ``groundcheck`` command).
"""

from groundcheck.cli import main

if __name__ == "__main__":
    main()
