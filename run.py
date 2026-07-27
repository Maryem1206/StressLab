import sys, os, signal, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "modules_src" / "credit_module"))


def _free_port(port: int) -> None:
    """Kill any process already listening on the given port (Windows)."""
    try:
        result = subprocess.check_output(
            f'netstat -ano | findstr ":{port}"', shell=True, text=True
        )
        current_pid = os.getpid()
        killed = set()
        for line in result.splitlines():
            if "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid != current_pid and pid not in killed:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed.add(pid)
                        print(f"  [run.py] Ancien processus tué : PID {pid}")
                    except OSError:
                        pass
    except subprocess.CalledProcessError:
        pass


from app import create_app
app = create_app()

if __name__ == "__main__":
    _free_port(5000)
    print("\n" + "="*55)
    print("  StressLab v2 — http://localhost:5000")
    print("  Pages: /upload  /dashboard  /transmission")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
