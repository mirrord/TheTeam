#!/usr/bin/env python3
"""
Setup script for TheTeam web interface.
Creates necessary directories and initializes the application.
"""

import sys
from pathlib import Path


def create_directory(path: Path, description: str):
    """Create a directory if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {description}: {path}")
    else:
        print(f"✓ {description} exists: {path}")


def main():
    """Run setup."""
    print("=" * 60)
    print("TheTeam Web Interface Setup")
    print("=" * 60)
    print()

    # Get project root
    project_root = Path.cwd()

    # Create necessary directories
    print("Creating required directories...")
    print()

    directories = [
        (project_root / "configs" / "agents", "Agent configurations"),
        (project_root / "configs" / "flowcharts", "Flowchart configurations"),
        (project_root / "configs" / "tools", "Tool configurations"),
        (project_root / "data" / "conversations", "Conversation storage"),
        (project_root / "src" / "theteam" / "static", "Static files"),
    ]

    for path, desc in directories:
        create_directory(path, desc)

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -e .")
    print("  2. Build frontend: cd frontend && npm install && npm run build")
    print("  3. Start server: theteam")
    print()
    print("For development mode:")
    print("  Terminal 1: theteam --debug")
    print("  Terminal 2: cd frontend && npm run dev")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {e}", file=sys.stderr)
        sys.exit(1)
