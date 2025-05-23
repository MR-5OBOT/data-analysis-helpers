import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def df_check(df: pd.DataFrame, required_columns: list[str]) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty.")
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def pacman_progress(current, total):
    """Displays a Pacman-style progress bar in the console"""
    print()
    bar_length = 30
    filled = int(round(bar_length * current / float(total)))
    bar = ">" * filled + "-" * (bar_length - filled)
    print(f"\r Progress: [{bar}] {current}/{total}", end="", flush=True)


def export_figure_to_pdf(plots_list):
    pdf_path = f"{datetime.datetime.now().strftime('%Y-%m-%d')}.pdf"
    with PdfPages(pdf_path) as pdf:
        plots = plots_list
        for func, args in plots:
            fig = func(*args)
            if fig is not None:
                pdf.savefig(fig)
            plt.close()
    return pdf_path


def fix_csv_format(input_file, output_file):
    """
    Converts a poorly formatted CSV (with spaces/tabs) into proper comma-separated format.
    Keeps timestamps (date + time) together as one field.
    """
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) >= 8:
                timestamp = f"{parts[0]} {parts[1]}"
                rest = parts[2:]
                clean_line = ",".join([timestamp] + rest)
            else:
                clean_line = ",".join(parts)

            outfile.write(clean_line + "\n")
