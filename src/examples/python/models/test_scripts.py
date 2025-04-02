from glob import glob
from subprocess import run


def test_scripts():
    for script in glob("scripts/*/*/*.py"):
        print(f"Testing {script}")

        process = run(["python3", script], capture_output=True)
        if process.returncode != 0:
            traceback = process.stderr.decode()
            print(f"Error trace: {traceback}\n")
        else:
            print("Success\n")


if __name__ == "__main__":
    test_scripts()
