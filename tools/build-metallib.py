import argparse
import os
import glob
import subprocess


def run_command(command):
    pass


def build(input_dir, output_dir, name):
    source_files = glob.glob(os.path.join(input_dir, "**/*.metal"), recursive=True)
    for source_file in source_files:
        output_file = os.path.join(output_dir, os.path.basename(source_file))
        command = f"xcrun -sdk macosx metal -c {source_file} -o {output_file}"
        run_command(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="cutenn.metallib")
    parser.add_argument("-i", "--input-dir", type=str, default="nn")
    parser.add_argument("-od", "--output-dir", type=str, default="out")

    args = parser.parse_args()
