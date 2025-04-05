import os
import subprocess

import pytest

# Pfad zum Projektverzeichnis
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

def run_deptry(project_dir):
    """
    Runs deptry to check for missing or unused dependencies and returns the output.
    """
    result = subprocess.run(
        ["deptry", project_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result


def test_deptry_no_issues():
    """
    Test to ensure deptry finds no issues with dependencies.
    """
    result = run_deptry(PROJECT_DIR)

    # Deptry gibt einen Rückgabewert von 0 zurück, wenn keine Probleme gefunden wurden
    assert result.returncode == 0, f"Deptry found issues:\n{result.stdout}\n{result.stderr}"


def test_deptry_missing_dependencies():
    """
    Test to ensure there are no missing dependencies.
    """
    result = run_deptry(PROJECT_DIR)

    # Analysiere die Ausgabe von deptry
    missing_dependencies = []
    for line in result.stdout.splitlines():
        if "Missing dependency:" in line:
            missing_dependencies.append(line)

    assert not missing_dependencies, f"Missing dependencies found:\n{missing_dependencies}"


def test_deptry_unused_dependencies():
    """
    Test to ensure there are no unused dependencies.
    """
    result = run_deptry(PROJECT_DIR)

    # Analysiere die Ausgabe von deptry
    unused_dependencies = []
    for line in result.stdout.splitlines():
        if "Unused dependency:" in line:
            unused_dependencies.append(line)

    assert not unused_dependencies, f"Unused dependencies found:\n{unused_dependencies}"