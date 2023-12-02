from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import uuid
import warnings


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    request_type_str = request.method
    if request_type_str == "GET":
        path = "static/baseimagenew.svg"
        return render_template("index.html", href=path)
    else:
        text = request.form["text"]
        random_string = uuid.uuid4().hex
        path = "app/static/" + random_string + ".svg"

        # user input
        np_arr = floatsome_to_np_array(text).reshape(1, -1)
        print("float to array done")
        pkl_filename = "app/TrainedModel/proj_pickle.pkl"
        with open(pkl_filename, "rb") as file:
            model = pickle.load(file)
            print("model loaded")
        plot_graphs(model=model, new_input_arr=np_arr, output_file=path)
        return render_template("index.html", href=path)


def floatsome_to_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(",") if is_float(x)])
    return floats.reshape(len(floats), 1)


df = pd.read_csv("app/clean_data.csv")


def plot_graphs(model, new_input_arr, output_file):
    fig = make_subplots(rows=1, cols=2)
    boxplot1 = go.Box(
        x=df["isFraud"], y=df["newBalanceOrig"], jitter=0.3, pointpos=-1.8
    )
    boxplot2 = go.Box(
        x=df["isFraud"], y=df["oldBalanceOrig"], jitter=0.3, pointpos=-1.8
    )
    fig.add_trace(boxplot1, row=1, col=2)
    fig.add_trace(boxplot2, row=1, col=1)
    fig.update_xaxes(title_text="Fraud", row=1, col=1)
    fig.update_xaxes(title_text="Fraud", row=1, col=2)
    fig.update_yaxes(title_text="New Balance Origin", row=1, col=2)
    fig.update_yaxes(title_text="Old Balance Origin", row=1, col=1)
    fig.update_xaxes(tickvals=[0, 1], ticktext=["False", "True"], row=1, col=1)
    fig.update_xaxes(tickvals=[0, 1], ticktext=["False", "True"], row=1, col=2)
    warnings.filterwarnings("ignore", category=UserWarning)
    new_preds = model.predict(new_input_arr)
    warnings.resetwarnings()
    new_balance = new_input_arr[:, 4]
    old_balance = new_input_arr[:, 3]

    fig.add_trace(
        go.Scatter(
            x=new_preds - 0.44,
            y=old_balance,
            mode="markers",
            name="predicted output",
            marker=dict(color="green", size=10),
            line=dict(color="green", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=new_preds - 0.44,
            y=new_balance,
            mode="markers",
            name="predicted output",
            marker=dict(color="orange", size=10),
            line=dict(color="orange", width=1),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500, width=1000, title_text="Variation in Fraud Status by Balance"
    )
    fig.write_image(output_file, engine="kaleido")
    fig.show()
