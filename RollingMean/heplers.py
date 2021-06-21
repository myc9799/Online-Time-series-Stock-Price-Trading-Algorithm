import pandas as pd

def get_missing_dates():

    filled_2019 = pd.read_csv("test_data/2019_with_filling.csv")
    filled_2019 = filled_2019.set_index('date').sort_index()
    blank_2019 = pd.read_csv("test_data/2019_with_blanks.csv")
    blank_2019 = blank_2019.set_index('date').sort_index()
    missing = list(set(list(filled_2019.index)).difference(set(list(blank_2019.index))))

    return missing
