import subprocess
# Run Core_detector.py and capture the output
result_detector = subprocess.run(['python', 'Core_detector.py'], capture_output=True, text=True)
output_detector = result_detector.stdout

# Run Core_recognizer.py with the obtained output from Core_detector.py
result_recognizer = subprocess.run(['python', 'Core_recognizer.py'], capture_output=True, text=True)
output_recognizer = result_recognizer.stdout

result_angular = subprocess.run(['python', 'angular_distance.py'], capture_output=True, text=True)
output_angular = result_angular.stdout

result_similarity_cv = subprocess.run(['python', 'final_similarity_cv.py'], capture_output=True, text=True)
output_similarity_cv = result_similarity_cv.stdout

print(output_similarity_cv)