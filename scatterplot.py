import matplotlib.pyplot as plt
import pandas as pd

def load_game_data(filename):
    data_frame = pd.read_csv(filename)
    data_frame.reindex(data_frame.episode.astype(int).sort_values().index)
    return data_frame


if __name__ == "__main__":
    file = "custom_minerals_random.csv"
    data = load_game_data(file)
    xValues = pd.Series(data['episode'])
    yValues = pd.Series(data['return'])
    plt.scatter(xValues, yValues)
    plt.title('Working Title')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward (Return)')
    plt.show()
