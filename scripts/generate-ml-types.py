import shutil

# CSV File
named_output = "csv"
source = "/mnt/code/artifacts/data.csv"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created CSV output")

# CSV File
named_output = "json"
source = "/mnt/code/artifacts/test.json"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created JSON output")

# PNG File
named_output = "png"
source = "/mnt/code/artifacts/plot.png"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created PNG output")

# JPEG File
named_output = "jpeg"
source = "/mnt/code/artifacts/plot.jpeg"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created JPEG output")

# Notebook File
named_output = "notebook"
source = "/mnt/code/artifacts/notebook.ipynb"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created Notebook output")

# MLFlow Mode
named_output = "mlflow_model"
source = "/mnt/code/artifacts/model"
dest = f"/workflow/outputs/{named_output}"
shutil.copytree(source, dest)
print("Created MLflow model output")

