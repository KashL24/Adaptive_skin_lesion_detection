import pandas as pd


class DifficultyScore:
    """
    Loads lesion difficulty information from a CSV file
    and provides difficulty information for a given image.
    """

    def __init__(self, csv_path):
        """
        Initialize the loader.

        Parameters:
            csv_path (str): Path to final_difficulty_scores.csv
        """
        self.csv_path = csv_path

        # Load the CSV into a DataFrame
        self.df = pd.read_csv(csv_path)

        # Convert the DataFrame into a dictionary for fast lookup
        self.difficulty_dict = self.df.set_index("image_name").to_dict("index")

    def get_difficulty(self, image_name):
        """
        Return the difficulty information for a given image.

        Parameters:
            image_name (str): Example -> ISIC_0012810.jpg

        Returns:
            dict containing:
                difficulty_score
                difficulty_label
                lesion_size
                border_unevenness
                boundary_clarity
        """

        if image_name not in self.difficulty_dict:
            raise ValueError(f"{image_name} not found in CSV.")

        return self.difficulty_dict[image_name]


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":

    loader = DifficultyScore("final_difficulty_scores.csv")

    info = loader.get_difficulty("ISIC_0012810.jpg")

    print(info)