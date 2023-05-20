import sys
import pickle
from sentence_transformers import SentenceTransformer
from data_loader import DataLoader


class TrainModel:
    def __init__(self):
        # Define the model name for sentence transformation
        self.model_name = 'all-MiniLM-L6-v2'

        # Initialize the sentence transformer model
        self.model = SentenceTransformer(self.model_name)
        
        self.loader = DataLoader()
    

    def train(self):
        # Extract the dataset path and destination path from the command-line arguments
        dataset_path = sys.argv[1]
        dest_path = sys.argv[2]
        
        # Load the dataset
        self.schools_df = self.loader.load_csv(dataset_path)

        # Clean the data in the temporary address column
        self.schools_df["name_address_concat"] = self.schools_df["name_address_concat"].apply(self.loader.clean_string)

        # Encode the cleaned addresses into embeddings using the model
        dataset_embeddings = self.model.encode(self.schools_df["name_address_concat"].tolist())

        # Save the embeddings to a pickle file
        with open(dest_path, "wb") as f:
            pickle.dump(dataset_embeddings, f)


# Create an instance of the TrainModel class and call the train() method
train = TrainModel()
train.train()
