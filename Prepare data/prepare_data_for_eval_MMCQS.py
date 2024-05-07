# Import libraries
import pandas as pd
import json

# Read the test csv file
df = pd.read_csv('split/test.csv')  

# Function to create a JSON object from each row of the DataFrame
def create_json_object(row):
    
    image_path = f"MedSumm-ECIR2024/{row['image_path']}"
    # Create the prompt using the Codemixed_Question
    prompt = f"Summarize this hinglish text as a detailed English question: '{row['Codemixed_Question']}'. " \
             "Question should be about 40-50 words and start with 'What'/'Why'/'Can'/'Is'/'Does'/'Should', " \
             "should NOT have first person address (example I/my), followed by a line starting with 'The image here shows' depicting visual cue from the image."
    # Return the JSON object
    return {
        "question_id": row.name,  # Using the row index as the unique identifier
        "image": image_path,
        "text": prompt
    }

# Apply the function to each row and create a list of JSON objects
json_objects = [create_json_object(row) for index, row in df.iterrows()]

# Write the JSON Lines to a file
with open('split/test_lambda.jsonl', 'w') as jsonl_file:  
    for json_object in json_objects:
        jsonl_file.write(json.dumps(json_object) + "\n")

