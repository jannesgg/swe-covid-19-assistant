from flask import Flask, request, render_template
#from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np
import pandas as pd
import tensorflow_text
import tensorflow_hub as hub
from googletrans import Translator

translator = Translator()

app = Flask(__name__)
app.jinja_env.filters["zip"] = zip

global model_save_path
global model
global df
global embedding_mat


# Find the full list of pre-trained standard models here:
# https://github.com/jannesgg/sentence-transformers#sentence-embeddings-using-bert

model_save_path = "../models/training_stsbenchmark_bert-2019-09-04_08-13-43"
#model = SentenceTransformer(model_save_path)
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

df = pd.read_csv("../data/corona_data.csv")
embedding_mat = model(list(df.question.values)).numpy()
#embedding_mat = model.encode(list(df.question.values))


@app.route("/")
def my_form():
    return render_template("hello.html")


@app.route("/", methods=["POST"])
def query_check():

    #query = model.encode([request.form["text"]])
    lang = request.form["lang"]
    query = model([request.form["text"]])
    query_mat = np.full((len(df), np.array(embedding_mat).shape[1]), query)
    cosim_matrix = 1 - distance.cdist(query_mat, embedding_mat, "cosine").diagonal()
    results = [
        (translator.translate(a, dest=lang).text, b)
        for a, b in zip(
            list(
                df.reset_index()
                .iloc[cosim_matrix.argsort()[-5:][::-1]]
                .answer.values
            ),
            np.round(cosim_matrix[cosim_matrix.argsort()[-5:][::-1]], 2),
        )
    ]
    return render_template("hello.html", results=results)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
