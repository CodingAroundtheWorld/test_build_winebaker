from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/matching")
def get_matching():
        #python
    import string
    from ftfy import fix_text
    import re
    from collections import Counter
    import math
    #data
    import pandas as pd
    import numpy as np

    # The only global variabel, which is used in text_to_vector
    WORD = re.compile(r"\w+")

    def get_cosine(vec1, vec2):
        '''creates the cosine similiarty within the wine_data (y) '''
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(text):
        ''' converts the text into counters for the cosine_similairty'''
        words = WORD.findall(text)
        return Counter(words)

    def output(dataframe, list_cosine):
        '''creates the wine recommendation using the list_cosine to create a similairty column in the dataframe.
        Sorts the dataframe based on the similairty, return the most similiar wine to wine_label
        '''
        dataframe["similarity"] = list_cosine
        dataframe.sort_values(["similarity"], ascending=False)
        wine_reco_df = dataframe.sort_values(["similarity"], ascending=False)
        wine_reco_df.drop(columns=["text_bottle"], inplace=True)
        return wine_reco_df.head(1)

    #from matching import main
    def wine_reco(X):
        ''' the main function of matching.py where we clean the data, perform cosine_similairty on it.
        The output is a data_frame with the most similar wine'''
        #this will be our y dataset
        wine_data = pd.read_csv("Data/wine_dataset.csv",
                    usecols=["vintage_name","vintage_id","region","wine_price","food_1","flavor_1","flavor_2",
                    "flavor_3","image_bootle","grapes","text_bottle"])

        # setting both wine_data and X as str
        wine_data["text_bottle"] = wine_data["text_bottle"].apply(str)
        X = X.apply(str)

        # cleaning X and the wine_data
        for punct in string.punctuation:
            wine_data["text_bottle"] = wine_data["text_bottle"].str.replace(punct, " ")
            X = X.str.replace(punct, " ")

        wine_data["text_bottle"] = wine_data["text_bottle"].apply(lambda x: x.replace('\n', ' ').lower())
        X = X.apply(lambda x: x.replace('\n', ' ').lower())

        # creating the output (y)
        y = wine_data["text_bottle"]

        # list used to store the cosine similiarty
        list_cosine = []

        # X[0] is the index of the wine_label, text1 used for the vector1 of the cosine_similairty
        text1 =  X[0]

        for row in wine_data['text_bottle']:
            text2 = row
            vector1 = text_to_vector(text1)
            vector2 = text_to_vector(text2)
            list_cosine.append(get_cosine(vector1, vector2))



        return output(wine_data, list_cosine)

        print(wine_data)
