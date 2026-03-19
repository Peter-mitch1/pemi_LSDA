ENSURE USING CONDA POWERSHELL PROMPT
Also ensure you 'cd' to the correct file directory at the start

Prompts in first terminal:
conda activate ml
mlflow ui

Prompts in second terminal:

# Check if it's set
echo $env:MLFLOW_RUN_ID
# Clear it
Remove-Item Env:MLFLOW_RUN_ID -ErrorAction SilentlyContinue

# Run the file
mlflow run . --experiment-name wind-power-model --env-manager local