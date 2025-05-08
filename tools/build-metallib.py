import argparse
import os
import glob
import subprocess
from pathlib import Path


def run_command(command, cwd=None):
    try:
        ret = subprocess.run(
            command, cwd=cwd, shell=False, check=True, capture_output=True, text=True
        )
        return ret.stdout, ret.stderr, ret.returncode
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with return code {e.returncode}: {e.stderr}"
        )


def build(name, input_dir, output_dir, cwd, flags):
    source_files = glob.glob(os.path.join(input_dir, "**/*.metal"), recursive=True)
    air_files = []
    for source_file in source_files:
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.air")
        air_files.append(output_file)
        print(output_file)
        command = [
            "xcrun",
            "metal",
            "-c",
            source_file,
            "-I",
            cwd,
            "-o",
            output_file,
            *flags,
        ]
        stdout, stderr, returncode = run_command(command)

        # print("{} {} {}".format(stdout, stderr, returncode))
    command = [
        "xcrun",
        "metallib",
        "-o",
        "{}/cutenn.metallib".format(output_dir),
        *air_files,
    ]

    stdout, stderr, returncode = run_command(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="cutenn.metallib")
    parser.add_argument("-id", "--input-dir", type=str, default="nn")
    parser.add_argument("-od", "--output-dir", type=str, default="out")
    parser.add_argument(
        "-f",
        "--flags",
        type=list,
        default=["-Wall", "-Wextra", "-fno-fast-math", "-Werror"],
    )

    args = parser.parse_args()
    name = args.name
    input_dir = args.input_dir
    output_dir = args.output_dir
    flags = args.flags

    cwd = Path(__file__).resolve().parent.parent
    print("cwd {}".format(cwd))
    build(name, input_dir, "{}/{}".format(cwd, output_dir), str(cwd), flags)
