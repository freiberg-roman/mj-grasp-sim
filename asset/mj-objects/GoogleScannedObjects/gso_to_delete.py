import os
import shutil


def main():
    with open(os.path.join(os.path.dirname(__file__), "gso_to_delete.txt"), "r") as f:
        lines = f.readlines()

    # check if all lines have a corresponding directory
    for l in lines:
        l = l.strip()
        path_to_check = os.path.join(os.path.dirname(__file__), l)
        if not os.path.exists(path_to_check):
            raise FileNotFoundError(f"Directory {path_to_check} does not exist")

        # delete all non empty directory
        shutil.rmtree(path_to_check)


if __name__ == "__main__":
    main()
