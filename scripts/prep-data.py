import pandas as pd


def read_input(input_name):
    input_location = f"/workflow/inputs/{input_name}"
    with open(input_location, "r") as file:
        contents = file.read()
        return contents


# Read input data. Inputs are stored in a blob at /workflow/inputs/<NAME OF INPUT>.
# For file inputs, the blob is the file input itself. 
# For primitive type inputs like strings, integer, booleans, etc, they are stored inside the blob and must be read in by opening the file.
named_input = "data_path"
csv_file = read_input(named_input)
df = pd.read_csv(csv_file)

# Process data
print(df)
print("Preparing the data")
df = df.drop('a', axis=1)
print(df)

# Write output. Outputs must be written to /workflow/outputs/<NAME OF OUTPUT> for it to be tracked.
named_output = "processed_data"
df.to_csv("/workflow/outputs/{}".format(named_output))

named_output2 = "processed_data2"
df.to_csv("/workflow/outputs/{}".format(named_output2))
