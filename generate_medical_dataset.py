import argparse
import json
import os
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

# If using sentence-transformers
from sentence_transformers import SentenceTransformer
# If using OpenAI embeddings (requires API key)
# from openai import OpenAI

from dataclasses import dataclass, asdict

@dataclass
class DataPoint:
    """Class for storing medical knowledge data points."""
    name: str
    description_type: str
    description: str
    Q: str = None
    A: str = None
    key_string: str = None
    extended_Q: str = None
    extended_A: str = None

    def __post_init__(self):
        if not self.Q:
            self.Q = f"What is the {self.description_type} of {self.name}?"
        if not self.A:
            self.A = f"The {self.description_type} of {self.name} is {self.description}."
        if not self.key_string:
            self.key_string = f"the {self.description_type} of {self.name}"


def save_datapoint(datapoint, file_path):
    """Save a datapoint to a file in JSON format."""
    with open(file_path, "a") as f:
        f.write(json.dumps(asdict(datapoint)) + "\n")


def parse_markdown_medical_facts(md_file_path):
    """Parse a markdown file with medical facts into a structured format.
    
    Expected markdown format:
    # Disease/Condition Name
    
    ## Description
    Text describing the disease/condition.
    
    ## Symptoms
    Text listing symptoms.
    
    ## Treatment
    Text describing treatments.
    
    ## Prognosis
    Text describing prognosis.
    """
    with open(md_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by main headings (entities)
    entity_blocks = re.split(r'(?m)^# ', content)[1:]  # Skip the first empty part
    
    datapoints = []
    
    for block in entity_blocks:
        lines = block.strip().split('\n')
        entity_name = lines[0].strip()
        
        # Find subheadings and their content
        subheadings = re.findall(r'(?m)^## (.*?)$(.*?)(?=^##|\Z)', block, re.DOTALL)
        
        for description_type, description_content in subheadings:
            description_type = description_type.strip().lower()
            description_content = description_content.strip()
            
            if description_content:  # Only create datapoint if there's content
                datapoint = DataPoint(
                    name=entity_name,
                    description_type=description_type,
                    description=description_content
                )
                datapoints.append(datapoint)
    
    return datapoints


def compute_embeddings_local(model_name, datapoints, part, output_file, batch_size=32):
    """Compute embeddings using SentenceTransformer locally."""
    print(f"Computing {part} embeddings using {model_name}...")
    
    all_elements = []
    for dp in datapoints:
        if part == "key_string":
            all_elements.append(dp.key_string)
        elif part == "description":
            all_elements.append(dp.description)
        else:
            raise ValueError(f"Part {part} not supported.")
    
    # Process in batches to avoid memory issues
    chunks = [all_elements[i:i+batch_size] for i in range(0, len(all_elements), batch_size)]
    
    model = SentenceTransformer(model_name)
    embeddings = []
    
    for chunk in tqdm(chunks):
        embd = model.encode(chunk, convert_to_numpy=True)
        embeddings.append(embd)
    
    embeddings = np.concatenate(embeddings, 0)
    assert len(embeddings) == len(all_elements)
    
    np.save(output_file, embeddings)
    return embeddings


def compute_embeddings_openai(api_key, datapoints, part, output_file, model="text-embedding-3-large"):
    """Compute embeddings using OpenAI API."""
    print(f"Computing {part} embeddings using OpenAI {model}...")
    
    client = OpenAI(api_key=api_key)
    
    all_elements = []
    for dp in datapoints:
        if part == "key_string":
            all_elements.append(dp.key_string)
        elif part == "description":
            all_elements.append(dp.description)
        else:
            raise ValueError(f"Part {part} not supported.")
    
    embeddings = []
    
    for text in tqdm(all_elements):
        response = client.embeddings.create(
            model=model,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(output_file, embeddings_array)
    return embeddings_array


def create_train_test_split(datapoints, key_embeds, value_embeds, split_ratio=0.8, output_dir="dataset"):
    """Split the dataset into training and test sets."""
    print("Creating train/test split...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate split index
    split_index = int(len(datapoints) * split_ratio)
    
    # Split data
    train_dataset = datapoints[:split_index]
    test_dataset = datapoints[split_index:]
    
    # Split embeddings
    train_key_embds = key_embeds[:split_index]
    test_key_embds = key_embeds[split_index:]
    train_value_embds = value_embeds[:split_index]
    test_value_embds = value_embeds[split_index:]
    
    # Save datasets
    train_file = os.path.join(output_dir, "train_medical_data.json")
    test_file = os.path.join(output_dir, "test_medical_data.json")
    
    # Save as JSONL
    with open(train_file, "w") as f:
        for dp in train_dataset:
            f.write(json.dumps(asdict(dp)) + "\n")
    
    with open(test_file, "w") as f:
        for dp in test_dataset:
            f.write(json.dumps(asdict(dp)) + "\n")
    
    # Save embeddings
    np.save(os.path.join(output_dir, "train_medical_data_embd_key.npy"), train_key_embds)
    np.save(os.path.join(output_dir, "test_medical_data_embd_key.npy"), test_key_embds)
    np.save(os.path.join(output_dir, "train_medical_data_embd_value.npy"), train_value_embds)
    np.save(os.path.join(output_dir, "test_medical_data_embd_value.npy"), test_value_embds)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


def main():
    parser = argparse.ArgumentParser(description='Generate a medical dataset for KBLaM')
    parser.add_argument('--markdown_file', type=str, required=True, help='Path to markdown file with medical facts')
    parser.add_argument('--output_dir', type=str, default='medical_dataset', help='Output directory')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2', 
                        choices=['all-MiniLM-L6-v2', 'openai'], 
                        help='Embedding model to use')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key (if using OpenAI embeddings)')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse markdown file
    print(f"Parsing markdown file: {args.markdown_file}")
    datapoints = parse_markdown_medical_facts(args.markdown_file)
    print(f"Extracted {len(datapoints)} datapoints")
    
    # Save raw datapoints
    raw_output_file = os.path.join(args.output_dir, "medical_data.json")
    print(f"Saving raw datapoints to {raw_output_file}")
    with open(raw_output_file, "w") as f:
        for dp in datapoints:
            f.write(json.dumps(asdict(dp)) + "\n")
    
    # Compute embeddings
    if args.embedding_model == 'openai':
        if not args.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI embeddings")
        
        key_embeds = compute_embeddings_openai(
            args.openai_api_key, datapoints, "key_string", 
            os.path.join(args.output_dir, "medical_data_OAI_embd_key.npy")
        )
        value_embeds = compute_embeddings_openai(
            args.openai_api_key, datapoints, "description",
            os.path.join(args.output_dir, "medical_data_OAI_embd_value.npy")
        )
    else:
        key_embeds = compute_embeddings_local(
            args.embedding_model, datapoints, "key_string",
            os.path.join(args.output_dir, "medical_data_embd_key.npy")
        )
        value_embeds = compute_embeddings_local(
            args.embedding_model, datapoints, "description",
            os.path.join(args.output_dir, "medical_data_embd_value.npy")
        )
    
    # Create train/test split
    create_train_test_split(datapoints, key_embeds, value_embeds, args.split_ratio, args.output_dir)
    
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()