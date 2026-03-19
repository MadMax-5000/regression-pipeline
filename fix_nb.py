import json

path = "y:/code/regression-pipeline/notebooks/06_hyperparameter_tuning.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i, line in enumerate(cell["source"]):
            if 'mlflow.set_tracking_uri("file:///Y:/code/regression-pipeline/mlruns")' in line:
                cell["source"][i] = line.replace('file:///Y:/code/regression-pipeline/mlruns', 'sqlite:///Y:/code/regression-pipeline/mlflow.db')
            if 'force mlflow to use the root project mlruns folder' in line:
                cell["source"][i] = line.replace('mlruns folder', 'sqlite db folder')

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
