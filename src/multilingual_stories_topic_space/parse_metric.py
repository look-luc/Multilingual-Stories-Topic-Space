import re
from pathlib import Path

import pandas as pd

line_pattern = re.compile(r"\[(.*?)\]\s+(.*?)\s+CCA Correlations:\s+\[(.*?)\]")
float_pattern = re.compile(r"np\.float64\((-?\d+\.\d+)\)")

rows = []
base_path = Path(".") # Adjust to your actual base path

file_path = base_path / "src/model/more_metrics.txt"

with open(file_path, "r") as file:
    for line in file:
        match = line_pattern.match(line.strip())
        if match:
            lang, model_pair, floats_str = match.groups()

            # Extract all float values from the bracketed string chunk
            floats = [float(v) for v in float_pattern.findall(floats_str)]

            if len(floats) == 3:
                rows.append({
                    "Language": lang,
                    "Comparison": model_pair,
                    "Dim_1": floats[0],
                    "Dim_2": floats[1],
                    "Dim_3": floats[2]
                })

df = pd.DataFrame(rows)
print(df.to_latex())
