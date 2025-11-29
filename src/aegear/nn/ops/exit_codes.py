"""
Exit codes for training and HPO workflows.

These codes are used to communicate specific failure modes
between the training script, container runtime, and HPO orchestrator.
"""

# Success
EXIT_SUCCESS = 0

# General training failure (model error, data error, etc.)
EXIT_TRAINING_FAILURE = 1

# CUDA unavailable despite being required/expected
# This indicates a machine-level issue that should trigger a retry
# on a different machine
EXIT_CUDA_UNAVAILABLE = 42


def get_exit_code_description(code: int) -> str:
    """Get human-readable description for an exit code."""
    descriptions = {
        EXIT_SUCCESS: "Success",
        EXIT_TRAINING_FAILURE: "Training failure",
        EXIT_CUDA_UNAVAILABLE: "CUDA unavailable (machine issue - should retry)",
    }
    return descriptions.get(code, f"Unknown exit code: {code}")
