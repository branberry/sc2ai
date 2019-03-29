import matplotlib.pyplot as plt
import pandas as pd

def load_gamedata(filename):
    dataframe = pd.read_csv(filename)
    dataframe.reindex(dataframe.episode.astype(int).sort_values().index)
    return dataframe

if __name__ == "__main__":

    file = "custom_minerals_random.csv"
    data = load_gamedata(file)
    plt.plot(data['return'])
    plt.ylabel('Cumulative Reward (Return)')
    plt.xlabel('Episode')
    plt.show()
