import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/loopscope/phase11_d9_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("d9_bootstrap", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapDeadlineTests(unittest.TestCase):
    def test_preparation_and_launcher_share_one_deadline_and_timeout_child_is_reaped(self):
        started = time.monotonic()
        deadline = started + .8
        # An earlier owned preparation stage consumes the same budget.
        bootstrap.run_step([sys.executable, "-c", "import time; time.sleep(.2)"], deadline)
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "pid"
            command = [sys.executable, "-c",
                "import os,time; from pathlib import Path; "
                f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(10)"]
            with self.assertRaises(subprocess.TimeoutExpired):
                bootstrap.run_step(command, deadline)
            pid = int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        self.assertLess(time.monotonic()-started, 2)
        with self.assertRaises(TimeoutError):
            bootstrap.remaining(deadline)
        # Formal runs use their supplied duration, without a 1740-second clamp.
        self.assertGreater(bootstrap.remaining(time.monotonic()+86400), 86390)


if __name__ == "__main__":
    unittest.main()
