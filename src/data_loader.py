import pandas as pd
import pickle
import re


class DataLoader:

    """
    The following class is used to load the data from the csv files, as well as the encoded pkl files.
    It cleans the data in dataframe by removing unwanted characters and converting the string to lowercase.
    """

    # Function to load the csv file into a pandas DataFrame, and clean the data.
    def load_csv(self, file_path: str) -> pd.DataFrame:
        schools_df = pd.read_csv(file_path)
        schools_df["name_address_concat"] = schools_df["Name"] + " " + schools_df["Address"]
        schools_df["name_address_concat"] = schools_df["name_address_concat"].apply(self.clean_string)
        return schools_df


    # Function to load the pkl file using pickle
    def load_pkl(self, file_path: str):
        with open(file_path, "rb") as f:
            return pickle.load(f)


    # Function to clean the data in the dataframe
    def clean_data(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = df[column].apply(self.clean_string)
        return df


    # Function to clean the string
    def clean_string(self, s: str) -> str:
        # Convert the string to lowercase and remove unwanted characters using regular expressions.
        # Remove periods, quotes, hyphens, brackets from string using regex
        s = s.lower().strip()
        s = re.sub(r'[\.\,\"\'\(\)\-]', '', s)
        s = re.sub(r'\b\D\b(\s+\b\D\b)+', lambda x: x.group(0).replace(' ', ''), s)
        return s