import pandas as pd
import numpy as np


def list_for_panda():
    csv_file = "./panda-default_stat.csv"
    df = pd.read_csv(csv_file)

    # Compute the estimated time of arrival (eta) for 1000 successful grasps.
    # For rows with 0 successful grasps, we use NaN to avoid division-by-zero.
    df["eta"] = np.where(
        df["number_successful_grasps"] > 0,
        df["total_time"] * 1000.0 / df["number_successful_grasps"],
        np.nan,
    )

    # Sort the DataFrame by 'eta' in ascending order
    df_sorted = df.sort_values(by="eta", ascending=True)

    df_selected = df_sorted[[
        "name", "number_successful_grasps", "total_time", "eta"]]

    # Print out the first 400 entries
    print(df_selected.head(400))

    # Extract the names from the first 400 entries
    first_400_names = df_selected.head(400)["name"]

    # Write the names to a text file, one name per line
    with open("names_panda.txt", "w") as file:
        for name in first_400_names:
            file.write(name + "\n")


def list_for_vx():
    csv_file = "./vx300-default_stat.csv"
    df = pd.read_csv(csv_file)

    # Compute the estimated time of arrival (eta) for 1000 successful grasps.
    # For rows with 0 successful grasps, we use NaN to avoid division-by-zero.
    df["eta"] = np.where(
        df["number_successful_grasps"] > 0,
        df["total_time"] * 1000.0 / df["number_successful_grasps"],
        np.nan,
    )

    # Sort the DataFrame by 'eta' in ascending order
    df_sorted = df.sort_values(by="eta", ascending=True)

    df_selected = df_sorted[[
        "name", "number_successful_grasps", "total_time", "eta"]]

    # Print out the first 400 entries
    print(df_selected.head(400))

    # Extract the names from the first 400 entries
    first_400_names = df_selected.head(400)["name"]

    # Write the names to a text file, one name per line
    with open("names_vx.txt", "w") as file:
        for name in first_400_names:
            file.write(name + "\n")


def list_for_shadow():
    csv_file = "./shadow-three_finger_pinch_stat.csv"
    df = pd.read_csv(csv_file)

    # Compute the estimated time of arrival (eta) for 1000 successful grasps.
    # For rows with 0 successful grasps, we use NaN to avoid division-by-zero.
    df["eta"] = np.where(
        df["number_successful_grasps"] > 0,
        df["total_time"] * 1000.0 / df["number_successful_grasps"],
        np.nan,
    )

    # Sort the DataFrame by 'eta' in ascending order
    df_sorted = df.sort_values(by="eta", ascending=True)

    df_selected = df_sorted[[
        "name", "number_successful_grasps", "total_time", "eta"]]

    # Print out the first 400 entries
    print(df_selected.head(400))

    # Extract the names from the first 400 entries
    first_400_names = df_selected.head(400)["name"]

    # Write the names to a text file, one name per line
    with open("names_shadow.txt", "w") as file:
        for name in first_400_names:
            file.write(name + "\n")


def find_intersection():
    # Read names from the panda file and convert them to a set
    with open("names_panda.txt", "r") as file:
        names_panda = {line.strip() for line in file if line.strip()}

    # Read names from the vx file and convert them to a set
    with open("names_vx.txt", "r") as file:
        names_vx = {line.strip() for line in file if line.strip()}

    # Read names from the shadow file and convert them to a set
    with open("names_shadow.txt", "r") as file:
        names_shadow = {line.strip() for line in file if line.strip()}

    # Calculate the intersection of the three sets
    common_names = names_panda & names_vx & names_shadow

    # Print the intersection (sorted alphabetically for clarity)
    print("Intersection of names from all three files:")
    for name in sorted(common_names):
        print(name)
    with open("names_common.txt", "w") as file:
        for name in common_names:
            file.write(name + "\n")
    print("Total common names:", len(common_names))


if __name__ == "__main__":
    # find_intersection()
