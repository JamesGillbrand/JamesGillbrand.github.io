import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playercareerstats, playergamelog, playerawards
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

derozan = playergamelog.PlayerGameLog(player_id="201942", season=2022, season_type_all_star="Regular Season")
derozan = derozan.get_data_frames()[0]


def MultiReg(X):

    X_train, X_test, y_train, y_test = train_test_split(X[["MIN", "FGA"]], X[["PTS"]], test_size=.3, random_state=32)
    reg = linear_model.LinearRegression().fit(X_train, y_train)

    predictions = reg.predict(X_test)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter(X[["FGA"]], X[["MIN"]], X[["PTS"]])

    # x = FGA, y = MIN, z = PTS
    coef = pd.DataFrame(zip(X.columns, reg.coef_))

    x = np.arange(0, 40, .25)
    y = np.arange(0, 70, .25)
    x, y = np.meshgrid(x, y)

    x_coef = coef[1][0][0]
    y_coef = coef[1][0][1]
    int = reg.intercept_
    z = int + x_coef * x + y_coef * y
    surf = ax.plot_wireframe(x, y, z)

    ax.set_xlabel("FGA")
    ax.set_ylabel("MIN")
    ax.set_zlabel("PTS")
    ax.set_title("FGA and MIN vs PTS")

    ax.set_zlim(0, 50)
    plt.show()

if __name__ == "__main__":
    MultiReg(derozan)
    print()
