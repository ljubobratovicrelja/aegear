"""
RunPod pod management utilities for training and HPO workflows.
"""
import sys
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Install it with: pip install requests")
    sys.exit(1)

class RunPodLauncher:
    """Manages RunPod pod lifecycle for training jobs."""
    RUNPOD_API_BASE = "https://api.runpod.io/graphql"
    DEFAULT_IMAGE = "docker.io/ljubobratovicrelja/aegear:latest"
    DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 5090"
    DEFAULT_VOLUME_SIZE = 10
    DEFAULT_CONTAINER_DISK_SIZE = 8

    def __init__(self, api_token: str, docker_username: Optional[str] = None, docker_pat: Optional[str] = None):
        self.api_token = api_token
        self.docker_username = docker_username
        self.docker_pat = docker_pat
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        })

    def _graphql_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        response = self.session.post(self.RUNPOD_API_BASE, json=payload)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise RuntimeError(f"GraphQL error: {result['errors']}")
        return result.get("data", {})

    def get_gpu_types(self) -> list:
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
            }
        }
        """
        try:
            result = self._graphql_query(query)
            return result.get("gpuTypes", [])
        except Exception as e:
            print(f"Warning: Could not fetch GPU types: {e}")
            return []

    def create_container_registry_auth(self) -> Optional[str]:
        if not self.docker_username or not self.docker_pat:
            return None
        query = """
        mutation SaveRegistryAuth($input: SaveRegistryAuthInput!) {
            saveRegistryAuth(input: $input) {
                id
                name
            }
        }
        """
        auth_name = f"dockerhub_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        variables = {
            "input": {
                "name": auth_name,
                "username": self.docker_username,
                "password": self.docker_pat
            }
        }
        try:
            result = self._graphql_query(query, variables)
            auth_id = result.get("saveRegistryAuth", {}).get("id")
            if auth_id:
                print(f"Created Docker Hub authentication: {auth_id}")
            return auth_id
        except Exception as e:
            print(f"Warning: Failed to create registry auth: {e}")
            print("  Proceeding without authentication (may hit Docker Hub rate limits)")
            return None

    def launch_pod(self, task_name: str, env_vars: Dict[str, str], gpu_type: str = DEFAULT_GPU_TYPE, gpu_count: int = 1, volume_size: int = DEFAULT_VOLUME_SIZE, container_disk_size: int = DEFAULT_CONTAINER_DISK_SIZE, image_name: str = DEFAULT_IMAGE) -> str:
        registry_auth_id = self.create_container_registry_auth()
        env_vars["RUNPOD_API_KEY"] = self.api_token
        query = """
        mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                desiredStatus
                imageName
                env
                machineId
                machine {
                    gpuDisplayName
                }
            }
        }
        """
        pod_env = [
            {"key": k, "value": str(v)}
            for k, v in env_vars.items()
        ]
        variables = {
            "input": {
                "name": task_name,
                "imageName": image_name,
                "gpuTypeId": gpu_type,
                "gpuCount": gpu_count,
                "volumeInGb": volume_size,
                "containerDiskInGb": container_disk_size,
                "volumeMountPath": "/workspace",
                "cloudType": "SECURE",
                "env": pod_env,
                "startSsh": False,
                "startJupyter": False,
            }
        }
        if registry_auth_id:
            variables["input"]["containerRegistryAuthId"] = registry_auth_id
        print(f"\nLaunching pod: {task_name}")
        print(f"   Image: {image_name}")
        print(f"   GPU: {gpu_type} x{gpu_count}")
        print(f"   Volume: {volume_size}GB, Container Disk: {container_disk_size}GB")
        try:
            result = self._graphql_query(query, variables)
        except Exception as e:
            print(f"\nFailed to create pod. Request details:")
            print(f"   GPU Type: {variables['input'].get('gpuTypeId', 'Not specified')}")
            print(f"   Cloud Type: {variables['input']['cloudType']}")
            raise
        pod = result.get("podFindAndDeployOnDemand", {})
        pod_id = pod.get("id")
        if not pod_id:
            raise RuntimeError("Failed to create pod - no ID returned")
        print(f"Pod created: {pod_id}")
        print(f"   Machine: {pod.get('machine', {}).get('gpuDisplayName', 'Unknown')}")
        print(f"   Self-termination: RUNPOD_POD_ID and RUNPOD_API_KEY are available in container")
        print(f"   Container will auto-terminate on training completion")
        return pod_id

    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        query = """
        query Pod($input: PodFilter!) {
            pod(input: $input) {
                id
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """
        variables = {"input": {"podId": pod_id}}
        result = self._graphql_query(query, variables)
        return result.get("pod", {})

    def get_pod_logs(self, pod_id: str, lines: int = 100) -> str:
        try:
            endpoint = f"https://api.runpod.io/v1/pods/{pod_id}/logs"
            response = self.session.get(endpoint, params={"lines": lines})
            if response.status_code == 200:
                return response.text
        except:
            pass
        return ""
    
    def get_pod_exit_code(self, pod_id: str) -> int:
        """
        Extract exit code from pod logs.
        
        Looks for exit code patterns in the last lines of logs.
        Returns 0 if no exit code found or if pod terminated successfully.
        Returns 42 if CUDA validation failed (machine issue).
        Returns 1 for other failures.
        
        Args:
            pod_id: The pod ID to check
            
        Returns:
            Exit code (0, 1, 42, etc.)
        """
        logs = self.get_pod_logs(pod_id, lines=200)
        
        if not logs:
            # No logs available, assume success if pod terminated
            return 0
        
        # Check for explicit exit code 42 (CUDA unavailable)
        if "exit 42" in logs.lower() or "exiting with code 42" in logs.lower():
            return 42
        
        # Check for CUDA validation failure messages
        if "CUDA VALIDATION FAILED" in logs:
            return 42
        
        if "DEVICE VALIDATION FAILED" in logs:
            return 42
        
        # Check for successful completion
        if "Training completed successfully" in logs:
            return 0
        
        # Check for training failure
        if "Training failed" in logs:
            return 1
        
        # Default to success if pod terminated cleanly
        return 0

    def terminate_pod(self, pod_id: str):
        query = """
        mutation PodTerminate($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        variables = {"input": {"podId": pod_id}}
        print(f"\nTerminating pod: {pod_id}")
        result = self._graphql_query(query, variables)
        if result.get("podTerminate"):
            print("Pod terminated successfully")
        else:
            print("Pod termination request sent (status unclear)")

    def monitor_pod(self, pod_id: str, check_interval: int = 60, timeout_hours: int = 24, auto_terminate: bool = True):
        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        print(f"\nMonitoring pod: {pod_id}")
        print(f"   Check interval: {check_interval}s")
        print(f"   Timeout: {timeout_hours}h")
        print(f"   Auto-terminate: {auto_terminate}")
        print("\n" + "="*60)
        try:
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"\nTimeout reached ({timeout_hours}h)")
                    if auto_terminate:
                        self.terminate_pod(pod_id)
                    return False
                try:
                    status = self.get_pod_status(pod_id)
                    if status is None or not status:
                        print(f"\n\nPod has been terminated (no longer found in API)")
                        print("   Training completed and pod self-terminated successfully!")
                        return True
                    desired_status = status.get("desiredStatus", "UNKNOWN")
                    runtime = status.get("runtime")
                    if desired_status in ["EXITED", "STOPPED", "TERMINATED"]:
                        print(f"\n\nPod stopped with status: {desired_status}")
                        return True
                    if runtime:
                        uptime = runtime.get("uptimeInSeconds", 0)
                        gpus = runtime.get("gpus", [])
                        print(f"\r[{time.strftime('%H:%M:%S')}] "
                              f"Status: {desired_status} | "
                              f"Uptime: {uptime//3600}h {(uptime%3600)//60}m | "
                              f"Elapsed: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m",
                              end="", flush=True)
                    else:
                        print(f"\r[{time.strftime('%H:%M:%S')}] "
                              f"Status: {desired_status} | "
                              f"Waiting for runtime...",
                              end="", flush=True)
                except Exception as e:
                    print(f"\n\nPod terminated (API query failed: {e})")
                    return True
                time.sleep(check_interval)
        except KeyboardInterrupt:
            print(f"\n\nMonitoring interrupted by user")
            if auto_terminate:
                print("Terminating pod...")
                self.terminate_pod(pod_id)
            return False
        finally:
            if auto_terminate:
                try:
                    self.terminate_pod(pod_id)
                except:
                    pass


class PodManager:
    """Manages RunPod pods with listing and termination capabilities."""
    
    RUNPOD_API_BASE = "https://api.runpod.io/graphql"
    
    # Status descriptions for user-friendly explanations
    STATUS_DESCRIPTIONS = {
        "RUNNING": "🟢 Pod is actively running",
        "PENDING": "🟡 Pod is being created/started",
        "EXITED": "⚫ Pod has stopped/exited",
        "STOPPED": "⚫ Pod is stopped",
        "TERMINATED": "⚫ Pod has been terminated",
        "CREATED": "🟡 Pod created but not started",
        "STARTING": "🟡 Pod is starting up",
    }
    
    def __init__(self, api_token: str):
        """Initialize the pod manager."""
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        })
    
    def _graphql_query(self, query: str, variables: Dict = None) -> Dict:
        """Execute a GraphQL query against RunPod API."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            response = self.session.post(self.RUNPOD_API_BASE, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "errors" in result:
                raise RuntimeError(f"GraphQL error: {result['errors']}")
            
            return result.get("data", {})
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
    
    def list_all_pods(self) -> List[Dict[str, Any]]:
        """List all pods in the account."""
        query = """
        query Pods {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    imageName
                    gpuCount
                    costPerHr
                    machine {
                        gpuDisplayName
                    }
                    runtime {
                        uptimeInSeconds
                        gpus {
                            gpuUtilPercent
                            memoryUtilPercent
                        }
                        ports {
                            privatePort
                            publicPort
                        }
                    }
                }
            }
        }
        """
        
        result = self._graphql_query(query)
        pods = result.get("myself", {}).get("pods", [])
        return pods
    
    def get_pod_details(self, pod_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific pod."""
        query = """
        query Pod($input: PodFilter!) {
            pod(input: $input) {
                id
                name
                desiredStatus
                imageName
                gpuCount
                costPerHr
                machine {
                    gpuDisplayName
                }
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
            }
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        result = self._graphql_query(query, variables)
        return result.get("pod", {})
    
    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate a specific pod."""
        query = """
        mutation PodTerminate($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        
        variables = {"input": {"podId": pod_id}}
        
        try:
            result = self._graphql_query(query, variables)
            return result.get("podTerminate", False)
        except Exception as e:
            print(f"  ❌ Failed to terminate pod {pod_id}: {e}")
            return False
    
    def format_uptime(self, seconds: int) -> str:
        """Format uptime in human-readable format."""
        delta = timedelta(seconds=seconds)
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        
        return " ".join(parts)
    
    def format_cost(self, cost_per_hr: float, uptime_seconds: int) -> str:
        """Calculate and format accumulated cost."""
        if cost_per_hr and uptime_seconds:
            hours = uptime_seconds / 3600
            total_cost = cost_per_hr * hours
            return f"${total_cost:.4f} (${cost_per_hr:.4f}/hr)"
        return "N/A"
    
    def calculate_cost(self, cost_per_hr: float, uptime_seconds: int) -> float:
        """Calculate accumulated cost."""
        if cost_per_hr and uptime_seconds:
            hours = uptime_seconds / 3600
            return cost_per_hr * hours
        return 0.0
    
    def print_pod_summary(self, pod: Dict[str, Any], detailed: bool = True):
        """Print formatted pod information."""
        pod_id = pod.get("id", "unknown")
        name = pod.get("name", "unnamed")
        status = pod.get("desiredStatus", "UNKNOWN")
        image = pod.get("imageName", "unknown")
        gpu_count = pod.get("gpuCount", 0)
        cost_per_hr = pod.get("costPerHr", 0)
        
        machine = pod.get("machine", {})
        gpu_name = machine.get("gpuDisplayName", "unknown GPU")
        
        runtime = pod.get("runtime")
        uptime_seconds = runtime.get("uptimeInSeconds", 0) if runtime else 0
        
        # Status icon and description
        status_desc = self.STATUS_DESCRIPTIONS.get(status, f"❓ Unknown status: {status}")
        
        print(f"\n{'='*80}")
        print(f"Pod: {name}")
        print(f"ID:  {pod_id}")
        print(f"{'='*80}")
        print(f"Status:  {status_desc}")
        print(f"GPU:     {gpu_name} x{gpu_count}")
        print(f"Image:   {image}")
        
        if uptime_seconds > 0:
            print(f"Uptime:  {self.format_uptime(uptime_seconds)}")
            print(f"Cost:    {self.format_cost(cost_per_hr, uptime_seconds)}")
        else:
            print(f"Uptime:  Not running")
            print(f"Cost:    ${cost_per_hr:.4f}/hr (when running)")
        
        if detailed and runtime:
            gpus = runtime.get("gpus", [])
            if gpus:
                print(f"\nGPU Utilization:")
                for i, gpu in enumerate(gpus):
                    gpu_util = gpu.get("gpuUtilPercent", 0)
                    mem_util = gpu.get("memoryUtilPercent", 0)
                    print(f"  GPU {i}: {gpu_util}% compute, {mem_util}% memory")
            
            ports = runtime.get("ports", [])
            if ports:
                print(f"\nExposed Ports:")
                for port in ports:
                    private = port.get("privatePort")
                    public = port.get("publicPort")
                    print(f"  {private} -> {public}")
    
    def list_pods_command(self):
        """List all pods with detailed information."""
        print("\n" + "="*80)
        print("RUNPOD - ALL PODS")
        print("="*80)
        
        try:
            pods = self.list_all_pods()
            
            if not pods:
                print("\n✓ No pods found")
                return
            
            print(f"\nFound {len(pods)} pod(s):")
            
            # Separate by status
            running = [p for p in pods if p.get("desiredStatus") in ["RUNNING", "STARTING"]]
            pending = [p for p in pods if p.get("desiredStatus") in ["PENDING", "CREATED"]]
            stopped = [p for p in pods if p.get("desiredStatus") in ["EXITED", "STOPPED", "TERMINATED"]]
            
            # Print running pods first
            if running:
                print(f"\n{'─'*80}")
                print(f"RUNNING PODS ({len(running)})")
                print(f"{'─'*80}")
                for pod in running:
                    self.print_pod_summary(pod, detailed=True)
            
            # Then pending
            if pending:
                print(f"\n{'─'*80}")
                print(f"PENDING PODS ({len(pending)})")
                print(f"{'─'*80}")
                for pod in pending:
                    self.print_pod_summary(pod, detailed=False)
            
            # Finally stopped
            if stopped:
                print(f"\n{'─'*80}")
                print(f"STOPPED PODS ({len(stopped)})")
                print(f"{'─'*80}")
                for pod in stopped:
                    self.print_pod_summary(pod, detailed=False)
            
            # Summary
            print(f"\n{'='*80}")
            print(f"SUMMARY: {len(running)} running, {len(pending)} pending, {len(stopped)} stopped")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n❌ Error listing pods: {e}")
            import traceback
            traceback.print_exc()
    
    def kill_pod_command(self, pod_id: str) -> bool:
        """Kill a specific pod."""
        print(f"\nTerminating pod: {pod_id}")
        
        try:
            # Get pod details first
            pod = self.get_pod_details(pod_id)
            if not pod:
                print(f"❌ Pod not found: {pod_id}")
                return False
            
            self.print_pod_summary(pod, detailed=False)
            
            # Confirm
            response = input("\nTerminate this pod? (y/N): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return False
            
            # Terminate
            success = self.terminate_pod(pod_id)
            if success:
                print(f"✓ Pod {pod_id} terminated successfully")
                return True
            else:
                print(f"❌ Failed to terminate pod {pod_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error terminating pod: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def kill_all_command(self, running_only: bool = False):
        """Kill all pods (with confirmation)."""
        print("\n" + "="*80)
        print("RUNPOD - KILL ALL PODS")
        print("="*80)
        
        try:
            pods = self.list_all_pods()
            
            if not pods:
                print("\n✓ No pods found")
                return
            
            # Filter pods based on running_only flag
            if running_only:
                pods_to_kill = [p for p in pods if p.get("desiredStatus") in ["RUNNING", "STARTING", "PENDING", "CREATED"]]
                print(f"\nFound {len(pods_to_kill)} active pod(s) to terminate:")
            else:
                pods_to_kill = pods
                print(f"\nFound {len(pods_to_kill)} pod(s) to terminate:")
            
            if not pods_to_kill:
                print("\n✓ No pods to terminate")
                return
            
            # Display all pods
            for pod in pods_to_kill:
                self.print_pod_summary(pod, detailed=False)
            
            # Calculate total cost
            total_cost = sum(
                self.calculate_cost(p.get("costPerHr", 0), 
                                  p.get("runtime", {}).get("uptimeInSeconds", 0) if p.get("runtime") else 0)
                for p in pods_to_kill
            )
            
            print(f"\n{'='*80}")
            print(f"Total accumulated cost: ${total_cost:.4f}")
            print(f"{'='*80}\n")
            
            # Confirm
            prompt = "⚠️  TERMINATE ALL THESE PODS? (y/N): "
            response = input(prompt).strip().lower()
            
            if response != 'y':
                print("\nCancelled. No pods were terminated.")
                return
            
            # Terminate all
            print("\nTerminating pods...")
            success_count = 0
            failed_count = 0
            
            for pod in pods_to_kill:
                pod_id = pod.get("id")
                name = pod.get("name", "unnamed")
                
                print(f"  Terminating {name} ({pod_id})...", end=" ")
                
                if self.terminate_pod(pod_id):
                    print("✓")
                    success_count += 1
                else:
                    print("❌")
                    failed_count += 1
            
            print(f"\n{'='*80}")
            print(f"Termination complete: {success_count} succeeded, {failed_count} failed")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n❌ Error in kill-all operation: {e}")
            import traceback
            traceback.print_exc()
