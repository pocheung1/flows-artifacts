import shutil

# PDF Files
named_output = "pdf"
source = "/mnt/code/artifacts/test.pdf"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created PDF output")

# sas7bdat Files
named_output = "sas7bdat"
source = "/mnt/code/artifacts/test.sas7bdat"
dest = f"/workflow/outputs/{named_output}"
shutil.copy(source, dest)
print("Created SAS7BDAT output")

