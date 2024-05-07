# Import libraries
import pandas as pd
import json
from uuid import uuid4

# Load the CSV file
df = pd.read_csv(r'split/train.csv')

# Initialize an empty list to store the JSON objects
json_objects = []

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    # Generate a unique ID
    unique_id = str(uuid4())

    # Format the image path
    image_file = row['image_path']
    image_path = f"MedSumm-ECIR2024/{image_file}"

    # Create the JSON object for this row
    json_object = {
        "id": unique_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\nSummarize this hinglish text as a detailed English question: '{row['Codemixed_Question']}'. Question should be about 40-50 words and start with 'What'/'Why'/'Can'/'Is'/'Does'/'Should', should NOT have first person address (example I/my), followed by a line starting with 'The image here shows' depicting visual cue from the image."
            },
            {
                "from": "gpt",
                "value": row['Question_Summ_Updates']
            }
        ]
    }

    # Add the JSON object to the list
    json_objects.append(json_object)

# Convert the list of JSON objects to a JSON string
json_output = json.dumps(json_objects, indent=2)

# Save the JSON string to a file in the same directory as csv
with open('split/train_lambda.json', 'w') as outfile:
    outfile.write(json_output)

