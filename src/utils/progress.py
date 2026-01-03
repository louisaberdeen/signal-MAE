"""
Progress tracking utilities for pipeline monitoring.
"""

from typing import Dict, Optional
import pandas as pd


class ProgressTracker:
    """
    Track and display pipeline progress.

    Provides:
    - Step-by-step progress tracking
    - Timing information
    - Status reporting (running, completed, failed)
    - Summary output
    """

    def __init__(self):
        self.steps: Dict[str, dict] = {}

    def start(self, step_name: str) -> None:
        """
        Mark a step as started.

        Args:
            step_name: Name of the step
        """
        self.steps[step_name] = {
            "status": "running",
            "start_time": pd.Timestamp.now()
        }
        print(f"\n{'='*60}")
        print(f"Starting: {step_name}")
        print(f"{'='*60}")

    def complete(self, step_name: str) -> None:
        """
        Mark a step as completed.

        Args:
            step_name: Name of the step
        """
        if step_name in self.steps:
            elapsed = pd.Timestamp.now() - self.steps[step_name]["start_time"]
            self.steps[step_name]["status"] = "completed"
            self.steps[step_name]["elapsed"] = elapsed
            print(f"\nCompleted: {step_name} (Elapsed: {elapsed.total_seconds():.2f}s)")
        else:
            print(f"Warning: Step '{step_name}' not found in tracker")

    def error(self, step_name: str, error_msg: str) -> None:
        """
        Mark a step as failed.

        Args:
            step_name: Name of the step
            error_msg: Error message
        """
        if step_name in self.steps:
            self.steps[step_name]["status"] = "failed"
            self.steps[step_name]["error"] = error_msg
            print(f"\nFailed: {step_name}")
            print(f"Error: {error_msg}")

    def summary(self) -> None:
        """Display summary of all steps."""
        print(f"\n{'='*60}")
        print("Pipeline Summary")
        print(f"{'='*60}")

        for step, info in self.steps.items():
            status = info["status"]
            elapsed = info.get("elapsed", "N/A")

            if elapsed != "N/A":
                elapsed = f"{elapsed.total_seconds():.2f}s"

            status_symbol = {
                "completed": "[OK]",
                "running": "[RUNNING]",
                "failed": "[FAILED]"
            }.get(status, "[?]")

            print(f"  {status_symbol} {step}: {status} ({elapsed})")

            if status == "failed" and "error" in info:
                print(f"      Error: {info['error']}")

    def get_elapsed(self, step_name: str) -> Optional[float]:
        """
        Get elapsed time for a step in seconds.

        Args:
            step_name: Name of the step

        Returns:
            Elapsed time in seconds, or None if not completed
        """
        if step_name not in self.steps:
            return None
        elapsed = self.steps[step_name].get("elapsed")
        if elapsed is None:
            return None
        return elapsed.total_seconds()
