import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def can_bind_port(port: int) -> bool | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            return True
    except PermissionError:
        return None
    except OSError:
        return False


def find_available_port(preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 20):
        available = can_bind_port(port)
        if available is True:
            return port
        if available is None:
            return preferred_port
    raise RuntimeError(
        f"No available port found from {preferred_port} to {preferred_port + 19}"
    )


def wait_for_tcp_port(process: subprocess.Popen, port: int, timeout: float = 25) -> bool:
    deadline = time.time() + timeout
    last_returncode = None

    while time.time() < deadline:
        last_returncode = process.poll()
        if last_returncode is not None:
            return False

        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except PermissionError:
            time.sleep(3)
            return process.poll() is None
        except OSError:
            time.sleep(0.5)

    return process.poll() is None and last_returncode is None


def start_fastapi(port: int, env: dict) -> tuple[subprocess.Popen, int]:
    for candidate in range(port, port + 20):
        available = can_bind_port(candidate)
        if available is False:
            print(f"[System] FastAPI port {candidate} is busy; trying next port...")
            continue

        print(f"[System] Starting FastAPI backend and folder monitor on port {candidate}...")
        process = subprocess.Popen([
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(candidate),
            "--no-access-log",
        ], env=env)
        if wait_for_tcp_port(process, candidate):
            return process, candidate
        print(f"[System] FastAPI did not stay up on port {candidate}; trying next port...")
    raise RuntimeError("FastAPI could not start on ports 8000-8019")


def start_streamlit(port: int, env: dict) -> tuple[subprocess.Popen, int]:
    for candidate in range(port, port + 20):
        available = can_bind_port(candidate)
        if available is False:
            print(f"[System] Streamlit port {candidate} is busy; trying next port...")
            continue

        print(f"[System] Starting Streamlit UI on port {candidate}...")
        process = subprocess.Popen([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/streamlit_app.py",
            "--server.port",
            str(candidate),
            "--server.headless",
            "true",
        ], env=env)
        if wait_for_tcp_port(process, candidate):
            return process, candidate
        print(f"[System] Streamlit did not stay up on port {candidate}; trying next port...")
    raise RuntimeError("Streamlit could not start on ports 8501-8520")


def start_services():
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)

    processes: list[subprocess.Popen] = []
    fastapi_port = int(os.getenv("FASTAPI_PORT", "8000"))
    streamlit_port = int(os.getenv("STREAMLIT_PORT", "8501"))
    fastapi_port = find_available_port(fastapi_port)
    streamlit_port = find_available_port(streamlit_port)
    env = os.environ.copy()

    try:
        print("=" * 60)
        print("Starting AI Invoice Auditor")
        print("=" * 60)

        fastapi_process, fastapi_port = start_fastapi(fastapi_port, env)
        processes.append(fastapi_process)
        env["INVOICE_API_URL"] = f"http://localhost:{fastapi_port}"

        streamlit_process, streamlit_port = start_streamlit(streamlit_port, env)
        processes.append(streamlit_process)

        print("\n" + "=" * 60)
        print("AI Invoice Auditor is online")
        print(f"Streamlit Dashboard : http://localhost:{streamlit_port}")
        print(f"Backend API         : http://localhost:{fastapi_port}")
        print("Auto monitor        : data/incoming")
        print("Press Ctrl+C to stop.")
        print("=" * 60 + "\n")

        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Service exited unexpectedly with code {process.returncode}: "
                        f"{process.args}"
                    )
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[System] Shutting down services...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[System] Shutdown complete.")


if __name__ == "__main__":
    start_services()
