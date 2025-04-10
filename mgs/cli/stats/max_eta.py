import pandas as pd
import numpy as np


def compute_eta(df):
    """
    Compute the estimated time (eta) for 1000 successful grasps.
    If number_successful_grasps is zero, assign NaN.
    """
    df["eta"] = np.where(
        df["number_successful_grasps"] > 0,
        df["total_time"] * 1000.0 / df["number_successful_grasps"],
        np.nan,
    )
    return df


def max_eta_for_gripper(csv_file, common_names):
    """
    Load the CSV file for a gripper, compute the eta,
    filter to include only those objects in common_names, and then
    return the object name and eta value with the maximum eta.
    """
    df = pd.read_csv(csv_file)
    df = compute_eta(df)

    # Filter the DataFrame to include only rows whose 'name' is in the common names set
    df_filtered = df[df["name"].isin(common_names)]

    # Drop rows where eta is NaN (in case there are divisions by zero)
    df_filtered = df_filtered.dropna(subset=["eta"])

    # If there are any rows left, get the one with the maximum eta
    if not df_filtered.empty:
        max_row = df_filtered.loc[df_filtered["eta"].idxmax()]
        object_name = max_row["name"]
        eta_value = max_row["eta"]
        return object_name, eta_value
    else:
        return None, None


def main():
    # Read the common names from names_common.txt
    with open("names_common.txt", "r") as f:
        common_names = {line.strip() for line in f if line.strip()}

    # Define the mapping of gripper names to their CSV file paths
    grippers = {
        "panda": "./panda-default_stat.csv",
        "vx": "./vx300-default_stat.csv",
        "shadow": "./shadow-three_finger_pinch_stat.csv",
    }

    # For each gripper, find and print the object (and its eta) with the maximum eta value
    for gripper, csv_file in grippers.items():
        object_name, eta_value = max_eta_for_gripper(csv_file, common_names)
        if object_name is not None:
            print(
                f"Gripper: {gripper} => Object with maximum eta: {object_name}, Eta: {eta_value}"
            )
        else:
            print(f"Gripper: {gripper} has no common objects with computed eta.")


if __name__ == "__main__":
    main()
