import sys
import pandas as pd
from collections import Counter

class GetCommonWords:
    def get_common_words(self):
        files = sys.argv[1:]  # Retrieve command line arguments excluding the script name
        schools = []
        for file in files:
            df = pd.read_csv(file)
            schools += df['Name'].tolist()

        words = [word.lower() for school in schools for word in school.split()]

        word_counts = Counter(words)

        most_common_words = word_counts.most_common(100)

        final_words_list = []

        for word, count in most_common_words:
            if len(word) <= 4 and len(word) > 1:
                final_words_list.append(word)

        return final_words_list



# Create an instance of the class with multiple file names
common_words = GetCommonWords()

# Call the method to get the common words list
words_list = common_words.get_common_words()
print(words_list)