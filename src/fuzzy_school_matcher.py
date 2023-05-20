from sentence_transformers import SentenceTransformer, util
import pandas as pd
from fuzzywuzzy import fuzz, process


class FuzzySchoolMatcher:

    """
    In the below class, we have used sentence transformer model to encode the school names
    and addresses into embeddings. Then we use the cosine similarity between the input query and
    the dataset embeddings to find the top k most similar embeddings. After that, use fuzzywuzzy
    to find the top 5 most similar strings from the top k matches, using the token_set_ratio scorer.
    """
    def __init__(self):

        # Define the model name for sentence transformation
        self.model_name = 'all-MiniLM-L6-v2'

        # Initialize the sentence transformer model
        self.model = SentenceTransformer(self.model_name)
    

    def fuzzy_using_ST(self,
                       query: str,
                       schools_df: pd.DataFrame,
                       schools_embeddings) -> list:
        

        # Encode the cleaned query into an embedding using the model
        query_embedding = self.model.encode(query)
        
        # Load the dataset embeddings from the provided file path

        # Compute cosine similarities between the input embedding and dataset embeddings
        similarities = util.pytorch_cos_sim(query_embedding, schools_embeddings)
            
        k = 25
        flattened_similarities = similarities.flatten()

        # Get the indices of the top k most similar embeddings
        sorted_indices = (-flattened_similarities).argsort()[:k]

        # Get the corresponding name and addresses from the dataset based on the indices
        top_k_strings = [schools_df["name_address_concat"].to_list()[idx] for idx in sorted_indices]

        # Perform fuzzy matching on the top k addresses
        top_5_matches = process.extractBests(query, top_k_strings, scorer=fuzz.token_set_ratio, limit=5)

        final_list = []

        # Extract the school name, address, and score for the top 5 matches and append them to the final list.
        for match in top_5_matches:
            name = schools_df.loc[schools_df["name_address_concat"] == match[0], "Name"].values[0]
            address = schools_df.loc[schools_df["name_address_concat"] == match[0], "Address"].values[0]
            score = match[1]
            final_list.append((name.title(), address.title(), score))
            
        return final_list

# # Create an instance of the SchoolMatcher class
# matcher = FuzzySchoolMatcher("cbse_affiliated_schools.csv")

# # Perform fuzzy matching using sentence transformers
# search_string = "delhi public school jodhpur"
# dataset_embeddings = "cbse_dataset_embeddings.pkl"
# results = matcher.fuzzy_using_ST(search_string, dataset_embeddings)

# # Print the matching results
# for result in results:
#     name, address, score = result
#     print(f"Name: {name}\nAddress: {address}\nScore: {score}\n")
