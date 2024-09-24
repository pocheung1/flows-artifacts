from time import sleep

import pandas as pd

# Read input data. Inputs are stored in a blob at /workflow/inputs/<NAME OF INPUT>.
# For file inputs, the blob is the file input itself. 
# For primitive type inputs like strings, integer, booleans, etc, they are stored inside the blob and must be read in by opening the file.
named_input = "processed_data"
data_path = "/workflow/inputs/{}".format(named_input)
df = pd.read_csv(data_path)

# Pretend like something is happening here to train the model
print("Training the model")
sleep(20)

# Write output. Outputs must be written to /workflow/outputs/<NAME OF OUTPUT> for it to be tracked.
named_output = "model"
df.to_csv("/workflow/outputs/{}".format(named_output))
