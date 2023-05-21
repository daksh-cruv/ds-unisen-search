from src.abbr_school_matcher import AbbrSchoolMatcher
from src.fuzzy_school_matcher import FuzzySchoolMatcher
import json
from src.data_loader import DataLoader
import pandas as pd


class SearchEngine(AbbrSchoolMatcher, FuzzySchoolMatcher):

    """
    This class loads the CSV and PKL files into memory and performs search based on the board name.
    It is reponsible for loading the data and selecting the dataset based on the board name.
    The 'search()' method performs search based on the word lengths in the query.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        if hasattr(self, "initialized"):
            return
        self.initialized = True

        self.loader = DataLoader()

        # Load the JSON file containing the paths to the CSV and PKL files
        self.json_file = open(r"data\input\board_file_paths.json", "r")
        self.json_data = json.load(self.json_file)
        self.board_data = self.json_data["boards"]
        self.board_data_holder = {}

        self.board_data_loader()

        # Initialize the AbbrSchoolMatcher and FuzzySchoolMatcher classes
        super().__init__()
        super(AbbrSchoolMatcher, self).__init__()
    

    def board_data_loader(self):
        
        # Iterate over the board data
        for board_name, board_info in self.board_data.items():
            csv_path = board_info["csv_path"]
            pkl_path = board_info["pkl_path"]

            # Load CSV file into a pandas DataFrame
            df = self.loader.load_csv(csv_path)

            # Load PKL file using pickle
            pkl_data = self.loader.load_pkl(pkl_path)

            # Add the loaded data to the board_data_holder dictionary
            self.board_data_holder[board_name] = {"csv_data": df, "pkl_data": pkl_data}
 

    # Function to select the dataset based on the board name
    def select_dataset(self, board: str) -> tuple:
        csv_data = self.board_data_holder[board]["csv_data"]
        pkl_data = self.board_data_holder[board]["pkl_data"]
        return csv_data, pkl_data


    def search(self, query: str, board: str) -> list:
        
        """
        Perform search based on word lengths in the query.
        if all words are less than or equal to 4 characters, perform abbreviation search
        if all words are greater than 4 characters, perform fuzzy search
        else perform both and combine the results.
        Fuzzy search requires a cleaned string, abbreviation search does not.
        """

        # Retrieve the dataframe and embeddings based on the board name
        schools_df, schools_embeddings = self.select_dataset(board=board.upper())
        words = query.split()
        if all(len(word) <= 4 for word in words):
            results = self.abbreviation_search(query=query, schools_df=schools_df)

        elif all(len(word) > 4 for word in words):
            query = self.loader.clean_string(query)
            results = self.fuzzy_search(query=query,
                                        schools_df=schools_df,
                                        schools_embeddings=schools_embeddings)
            
        else:
            abbreviation_results = self.abbreviation_search(query=query, schools_df=schools_df)
            query = self.loader.clean_string(query)
            fuzzy_results = self.fuzzy_search(query=query,
                                              schools_df=schools_df,
                                              schools_embeddings=schools_embeddings)

            # Count the number of words with length <= 4 and > 4
            count_short_words = sum(1 for word in words if len(word) <= 4)
            count_long_words = sum(1 for word in words if len(word) > 4)

            """
            Adjust the score based on the count of short and long words.
            If the count of short words is less than the count of long words, adjust the score of fuzzy results,
            else adjust the score of abbreviation results.
            """
            if count_short_words < count_long_words:
                results = [(name, address, score + 10) if score is not None else None for name, address, score in fuzzy_results] + abbreviation_results
            else:
                results = fuzzy_results + [(name, address, score + 10) if score is not None else None for name, address, score in abbreviation_results]
                # results = fuzzy_results + abbreviation_results
        
        # Filter out non-null results, sort by score, and return
        non_null_results = [result for result in results if result is not None]
        non_null_results.sort(key=lambda x: x[2], reverse=True)
        return non_null_results


    def fuzzy_search(self,
                     query: str,
                     schools_df: pd.DataFrame,
                     schools_embeddings) -> list:

        # Perform fuzzy search using FuzzySchoolMatcher's fuzzy_using_ST method
        return super().fuzzy_using_ST(query=query,
                                      schools_df=schools_df,
                                      schools_embeddings=schools_embeddings)


    def abbreviation_search(self,
                            query: str,
                            schools_df = pd.DataFrame) -> list:

        # Perform abbreviation search using AbbrSchoolMatcher's abbreviation_search method
        return super().abbreviation_search(query=query,schools_df=schools_df)
