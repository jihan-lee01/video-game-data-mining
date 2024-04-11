import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler

class DataPreprocessor:
    def __init__(self, path: str):
        self.df = pd.read_csv(path, index_col=0)

    def get_floating_point_columns(self):
        return ["Length.All PlayStyles.Average", "Length.All PlayStyles.Leisure", "Length.All PlayStyles.Median", "Length.All PlayStyles.Polled",
                            "Length.All PlayStyles.Rushed", "Length.Completionists.Average", "Length.Completionists.Leisure", "Length.Completionists.Median",
                            "Length.Completionists.Polled", "Length.Completionists.Rushed", "Length.Main + Extras.Average", "Length.Main + Extras.Leisure",
                            "Length.Main + Extras.Median", "Length.Main + Extras.Polled", "Length.Main + Extras.Rushed", "Length.Main Story.Average",
                            "Length.Main Story.Leisure", "Length.Main Story.Median", "Length.Main Story.Polled", "Length.Main Story.Rushed", "Metrics.Sales"
                            ,"Metrics.Used Price"]

    def fill_missing_data_by_avg(self, column_names: list = None):
        if column_names is None:
            column_names = self.get_floating_point_columns()

        for c in column_names:
            self.fill_missing_data_by_avg_for_column(c)

    def fill_missing_data_by_avg_for_column(self, column_name: str):
        if not column_name in self.df.columns:
            raise AttributeError("No such column: " + column_name)

        self.df[column_name] = self.df[column_name].replace(0, np.nan).fillna(self.df[column_name].mean())

    # you can set your own column names
    def drop_useless_columns(self, column_names: list = None):
        if column_names is None:
            # they are all true
            column_names = ["Features.Handheld?", "Features.Multiplatform?", "Features.Online?", "Metadata.Licensed?", "Metadata.Sequel?", "Release.Re-release?"]

        self.df = self.df.drop(column_names, axis='columns')

    def split_genres_by_comma(self):
        self.df['Metadata.Genres'] = self.df['Metadata.Genres'].str.split(',')
        self.df = self.df.explode('Metadata.Genres')
        # Clean and transform data
        self.df = self.df.drop_duplicates()
        self.df = self.df.reset_index(drop=True)

    def label_encoding(self, column_names: list = None):
        if column_names is None:
            # they are all true
            column_names = ["Metadata.Genres", "Metadata.Publishers", "Release.Console", "Release.Rating"]

        for column_name in column_names:
            l1 = LabelEncoder()
            l1.fit(self.df[column_name])
            self.df[column_name] = l1.transform(self.df[column_name])

    def normalization(self, column_names: list = None):
        if column_names is None:
            # they are all true
            column_names = self.get_floating_point_columns()

        norm = Normalizer()
        self.df[column_names] = norm.fit_transform(self.df[column_names])

    def standard_scaling(self, column_names: list = None):
        if column_names is None:
            # they are all true
            column_names = self.get_floating_point_columns()

        ss = StandardScaler()
        self.df[column_names] = ss.fit_transform(self.df[column_names])

    def preprocess(self):
        # you can modify the process here
        self.fill_missing_data_by_avg()
        self.drop_useless_columns()
        self.split_genres_by_comma()
        self.label_encoding()
        self.standard_scaling()
        self.normalization()

if __name__ == '__main__':
    # data = pd.read_csv('../data/video_games.csv')
    # print(data.head(10))
    preprocessor = DataPreprocessor('../data/video_games.csv')
    preprocessor.preprocess()
    print(preprocessor.df)
    # cleaned_data = preprocessor.df()
    # return cleaned_data