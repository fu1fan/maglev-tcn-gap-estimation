import colorama
import os
import pandas as pd

class DataModule:
    def __init__(self, name: str, unit: str, header: str, k: float, offset: float):
        self.name = name
        self.unit = unit
        self.header = header
        self.k = k
        self.offset = offset

AirGap = DataModule("AirGap", "mm" , "design_1_i/system_ila_0/inst/probe1[11:0]", -0.0076923076923076923076923076923077, 15.23)
B = DataModule("B", "mt", "design_1_i/system_ila_0/inst/probe2[11:0]", 0.1567, -9.1492)
Force = DataModule("Force", "mt", "design_1_i/system_ila_0/inst/probe3[11:0]", 1.0, 1.0)
Voltage = DataModule("Voltage", "%", "design_1_i/system_ila_0/inst/probe4[15:0]", 0.0001, 0)
CurrentSmallSig = DataModule("CurrentSmallSig", "", "design_1_i/system_ila_0/inst/probe5[11:0]", 1, 0)
Current = DataModule("Current", "A", "design_1_i/system_ila_0/inst/probe6[11:0]", 0.0123, 0.1387)
Columns = [AirGap, B, Force, Voltage, CurrentSmallSig, Current]

def raw2csv(input_path: str, output_path: str):
    data = pd.read_csv(input_path)
    new = pd.DataFrame()
    for module in Columns:
        if module.header in data.columns:
            raw_bipolar = data[[module.header]]
            new[[module.name]] = raw_bipolar
            new[[f"{module.name}({module.unit})"]] = raw_bipolar * module.k + module.offset
        else:
            print(colorama.Fore.RED + "ERROR: {module.header} not found in the CSV file.")
    new.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw_dir = "data/raw"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    with os.scandir(raw_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".csv"):
                out_path = f"{out_dir}/{entry.name[:-4]}_processed.csv"
                if os.path.exists(out_path):
                    print(colorama.Fore.YELLOW + f"Dataset exist: {entry.name}, skipping...")
                    continue
                else:
                    in_path = f"{raw_dir}/{entry.name}"
                    print(f"processing: {in_path}")
                    raw2csv(in_path, out_path)
                    print(f"Done. Written to {out_path}")

