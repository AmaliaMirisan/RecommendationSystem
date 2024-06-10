import yaml

def txt_to_yaml(txt_file_path, yaml_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Prepare the structure for the YAML file
    conversations = []
    for line in lines:
        # Split the line into user input and bot response
        parts = line.strip().split('\t')
        if len(parts) == 2:
            conversations.append(parts)

    # Create the YAML structure
    data = {"conversations": conversations}

    # Write the structured data to a YAML file
    with open(yaml_file_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(data, yaml_file, allow_unicode=True, default_flow_style=False)


import csv
import yaml


def csv_to_yaml(csv_file_path, yaml_file_path):
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(csv_file)

        # Prepare the structure for the YAML file
        conversations = []

        # Read each row in the CSV
        for row in csv_reader:
            # Each row contains 'question' and 'answer' columns
            if 'question' in row and 'answer' in row:
                conversations.append([row['question'], row['answer']])

    # Create the YAML structure
    data = {"conversations": conversations}

    # Write the structured data to a YAML file
    with open(yaml_file_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(data, yaml_file, allow_unicode=True, default_flow_style=False)
# Example usage
csv_to_yaml('C:\\Users\\miris\\Desktop\\lic\\chatBot\\prepare_texts\\Conversation.csv', 'C:\\Users\\miris\\Desktop\\lic\\chatBot\\training_texts\\conversation.yml')
#csv_to_yaml('C:\\Users\\miris\\Desktop\\lic\\chatBot\\prepare_texts\\Conversation.csv', 'C:\\Users\\miris\\Desktop\\lic\\chatBot\\training_texts\\conversation.yml')
#txt_to_yaml('C:\\Users\\miris\\Desktop\\lic\\chatBot\\prepare_texts\\dialogs.txt', '/training_texts/chat_data.yml')
