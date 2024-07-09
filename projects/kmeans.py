import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

df = pd.read_csv("FG3M_FTM_FGA.csv")

def plot(threes_ftm):
    plt.scatter(threes_ftm[["FTM"]], threes_ftm[["FG3M"]], c=threes_ftm[["FGA"]].values.tolist())
    plt.colorbar()

    plt.xlabel("FTM")
    plt.ylabel("FG3M")
    plt.title("FTM v FG3M")

    #plt.show()

def linearReg(df):
    model = LinearRegression()
    model.fit(df.FTM.values.reshape(-1,1), df.FG3M.values.reshape(-1,1))

    X = df.FTM.values.reshape(-1, 1)
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1,1))

    plt.plot(x_range, y_range)

    plt.show()

def kMeans(df):
    kmeans = KMeans(n_clusters=4)
    label = kmeans.fit_predict(df[["FTM", "FG3M"]])
    plt.scatter(df.FTM, df.FG3M, c=label)

    plt.xlabel("FTM")
    plt.ylabel("FG3M")
    plt.title("FTM v FG3M")

    u_labels = np.unique(label)


if __name__ == "__main__":
    kMeans(df)
    linearReg(df)
