# Import libraries
import json
from uuid import uuid4

# Load the question and answer JSON files
with open(r'OpenEnded_mscoco_train2014.json', 'r') as file:
    questions_data = json.load(file)

with open(r'Annotations_Train_mscoco\mscoco_train2014_annotations.json', 'r') as file:
    answers_data = json.load(file)

# Function to construct image path
def construct_image_path(image_id):
    formatted_id = f"{image_id:012d}"
    return f"/content/train2014/train2014/COCO_train2014_{formatted_id}.jpg"
    # return f"COCO_train2014_000000{image_id}.jpg"  

# Build a lookup dictionary for answers by question_id
answers_dict = {item['question_id']: item for item in answers_data['annotations']}

# Combine the data into the required format
combined_data = []
for question in questions_data['questions']:
    image_path = construct_image_path(question['image_id'])
    answer_entry = answers_dict.get(question['question_id'])

    unique_id = str(uuid4())
    
    answer = answer_entry['multiple_choice_answer'] if answer_entry else "No answer found"
    
    combined_data.append({
        "id": unique_id,
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question['question']}"
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    })

# Save the combined data to a JSON file
with open('combined_mcvqa_data.json', 'w') as file:
    json.dump(combined_data, file, indent=4)
