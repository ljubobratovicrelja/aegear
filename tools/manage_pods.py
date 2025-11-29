#!/usr/bin/env python3
"""
RunPod Management Tool

Manage RunPod pods: list all pods, view detailed status, and bulk terminate.

Usage:
    # List all pods with detailed status
    python manage_pods.py list
    
    # Kill all pods (with confirmation)
    python manage_pods.py kill-all
    
    # Kill specific pod
    python manage_pods.py kill <pod_id>
    
    # Kill all running pods only
    python manage_pods.py kill-running

Environment variables:
    RUNPOD_API_TOKEN: RunPod API token (required)
"""

import os
import sys
import argparse

from aegear.nn.ops import PodManager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RunPod Pod Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all pods with detailed status
  python manage_pods.py list
  
  # Kill all pods (with confirmation)
  python manage_pods.py kill-all
  
  # Kill only running/active pods
  python manage_pods.py kill-running
  
  # Kill specific pod
  python manage_pods.py kill <pod_id>

Environment Variables:
  RUNPOD_API_TOKEN    RunPod API token (required)
        """
    )
    
    parser.add_argument(
        "command",
        choices=["list", "kill-all", "kill-running", "kill"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "pod_id",
        nargs="?",
        help="Pod ID (required for 'kill' command)"
    )
    
    parser.add_argument(
        "--api-token",
        type=str,
        default=os.getenv("RUNPOD_API_TOKEN"),
        help="RunPod API token (default: $RUNPOD_API_TOKEN)"
    )
    
    args = parser.parse_args()
    
    # Validate API token
    if not args.api_token:
        print("❌ Error: RunPod API token not provided.")
        print("  Set RUNPOD_API_TOKEN environment variable or use --api-token")
        return 1
    
    # Validate kill command
    if args.command == "kill" and not args.pod_id:
        print("❌ Error: pod_id required for 'kill' command")
        print("  Usage: python manage_pods.py kill <pod_id>")
        return 1
    
    # Initialize manager
    manager = PodManager(args.api_token)
    
    # Execute command
    try:
        if args.command == "list":
            manager.list_pods_command()
        elif args.command == "kill-all":
            manager.kill_all_command(running_only=False)
        elif args.command == "kill-running":
            manager.kill_all_command(running_only=True)
        elif args.command == "kill":
            success = manager.kill_pod_command(args.pod_id)
            return 0 if success else 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
