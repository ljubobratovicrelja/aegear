#!/usr/bin/env python3
"""Check status of a specific RunPod pod."""
import os
import sys
from launch_runpod_training import RunPodLauncher

if len(sys.argv) < 2:
    print("Usage: python check_pod_status.py <pod_id>")
    sys.exit(1)

pod_id = sys.argv[1]
api_token = os.getenv('RUNPOD_API_TOKEN')

if not api_token:
    print("Error: RUNPOD_API_TOKEN not set")
    sys.exit(1)

launcher = RunPodLauncher(api_token)
status = launcher.get_pod_status(pod_id)

print(f"\nPod: {pod_id}")
print(f"Desired Status: {status.get('desiredStatus', 'UNKNOWN')}")

runtime = status.get('runtime')
if runtime:
    print(f"Uptime: {runtime.get('uptimeInSeconds', 0)}s")
    ports = runtime.get('ports', [])
    if ports:
        print("Ports:")
        for port in ports:
            print(f"  {port.get('privatePort')} -> {port.get('publicPort')}")
else:
    print("Runtime: Not started yet")
