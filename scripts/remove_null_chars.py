from pathlib import Path

report = Path("reports/report_2024-01-24_094330.csv")
with report.open("r") as f:
    data = f.read()
    data = data.replace("\x00", "")
with report.open("w") as f:
    f.write(data)
