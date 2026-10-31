"""Strict numeric CLI parsing; no BPF privileges or GPU are needed."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BPF_BIN", Path(os.environ.get("REQSCOPE_BIN", ROOT / "build/reqscope")).with_name("reqscope-bpf")))


@unittest.skipUnless(BIN.is_file(), "build reqscope-bpf first")
class BpfCliTests(unittest.TestCase):
    def test_invalid_numbers_are_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            cases = [("--pid", str(os.getpid()) + "suffix"), ("--pid", "4294967297"),
                     ("--pid", "-1"), ("--pid", "+1"), ("--pid", " 1"),
                     ("--poll-ms", "4294967297"), ("--poll-ms", "1tail"),
                     ("--ring-seconds", "30tail"), ("--duration-seconds", "1tail"),
                     ("--duration-seconds", "18446744073709551615"),
                     ("--duration-seconds", "18446744073709551616")]
            for option, value in cases:
                with self.subTest(option=option, value=value):
                    result = subprocess.run([str(BIN), "--pid", str(os.getpid()), "--output", str(output),
                                             "--allow-schedstat", "--duration-seconds", "1", option, value],
                                            capture_output=True, text=True, timeout=5)
                    self.assertNotEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(output.exists(), result.stderr)


if __name__ == "__main__":
    unittest.main()
