import json
import logging
import pathlib
import re
from typing import Any

import pandas as pd
from tabulate import tabulate


def parse_benchmark_json():
    records = []
    for result_json_path in pathlib.Path("autogen_benchmark_results").glob("*.json"):
        logging.info(f"Reading: {result_json_path}")
        records.append(json.loads(result_json_path.read_text()))
    return pd.DataFrame(records)


def parse_benchmark_data(content: str):
    lines = content.splitlines()
    records: dict[str, Any] = []

    for line in lines:
        if not line.startswith("20"):
            logging.warning(f"Skipping line: {line}")
            continue

        # Split the line at "repro: python"
        main_part, *repro_part = line.split("repro: python")

        # Join the repro part back together in case it was split
        repro_command = "python " + " ".join(repro_part).strip() if repro_part else ""

        parts = main_part.split(",")
        if len(parts) >= 5:
            date = parts[0].strip()
            tok_s = float(re.search(r"tok/s=\s*(\d+\.\d+)", parts[1]).group(1))
            mem_s = float(re.search(r"mem/s=\s*(\d+\.\d+)", parts[2]).group(1))
            peak_mem = float(re.search(r"peak_mem=\s*(\d+\.\d+)", parts[3]).group(1))
            model_size = float(
                re.search(r"model_size=\s*(\d+\.\d+)", parts[4]).group(1)
            )

            quant = re.search(r"quant: (\S+)", main_part).group(1)
            mod = re.search(r"mod: (\S+)", main_part).group(1)
            kv_quant = re.search(r"kv_quant: (\S+)", main_part).group(1)
            model_compiled = re.search(r"compile: (\S+)", main_part).group(1)
            prefill_compiled = re.search(r"compile_prefill: (\S+)", main_part).group(1)
            dtype = re.search(r"dtype: (\S+)", main_part).group(1)

            # Extract model name from the 'mod' field
            model = mod.replace("-", "_").lower()

            records.append(
                {
                    "model": model,
                    "tok/s": tok_s,
                    "mem/s (GB/s)": mem_s,
                    "peak_mem (GB)": peak_mem,
                    "model_size (GB)": model_size,
                    "quant": quant,
                    "kv_quant": kv_quant,
                    "compile": model_compiled,
                    "compile_prefill": prefill_compiled,
                    "dtype": dtype,
                    "repro_command": repro_command,
                }
            )

    df = pd.DataFrame(records)
    df = df.astype(
        {
            "model": "category",
            "tok/s": "float64",
            "mem/s (GB/s)": "float64",
            "peak_mem (GB)": "float64",
            "model_size (GB)": "float64",
            "quant": "category",
            "kv_quant": "bool",
            "compile": "bool",
            "compile_prefill": "bool",
            "dtype": "category",
            "repro_command": "string",
        }
    )
    return df


def render_dataframe(df, max_width=None):
    if max_width:
        with pd.option_context("display.max_colwidth", max_width):
            print(
                tabulate(
                    df,
                    headers="keys",
                    tablefmt="pretty",
                    floatfmt=".2f",
                    showindex=False,
                )
            )
    else:
        print(
            tabulate(
                df, headers="keys", tablefmt="pretty", floatfmt=".2f", showindex=False
            )
        )


logging.basicConfig(level=logging.INFO)

# Parse the content
# content = pathlib.Path("benchmark_results.txt").read_text()
# df = parse_benchmark_data(content)

df = parse_benchmark_json()

# Display the DataFrame (excluding the repro_command for readability)
print("Parsed Benchmark Data (excluding repro_command):")
render_dataframe(df.drop(columns=["repro_command"]))

# # Display some basic statistics
# print("\nBasic Statistics:")
# print(df.drop(columns=["repro_command"]).describe())

# # Group by model and quantization method, and calculate mean performance metrics
# print("\nMean Performance Metrics by Model and Quantization:")
# grouped_stats = df.groupby(["model", "quant"])[
#     ["tok/s", "mem/s (GB/s)", "peak_mem (GB)", "model_size (GB)"]
# ].mean()
# render_dataframe(grouped_stats.reset_index())

# # Display reproduction commands separately
# print("\nReproduction Commands:")
# render_dataframe(df[["model", "quant", "repro_command"]], max_width=80)
