import subprocess

python_files = ['Core_detector.py', 'Core_recognizer.py']
outputs = []

for file in python_files:
    # Run the Python file and capture the output
    result = subprocess.run(['python', file], capture_output=True, text=True)
    output = result.stdout
    outputs.append(output)

# Get the combined output of all files
combined_output = ''.join(outputs)

# Pass the combined output as input to the second file
subprocess.run(['python', 'Core_recognizer.py'], input=combined_output, text=True)

# Run another_file.py with the obtained values
result_another_file = subprocess.run(['python', 'find_person_db.py', combined_output], capture_output=True, text=True)
output_another_file = result_another_file.stdout