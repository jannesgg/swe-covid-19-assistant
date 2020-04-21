import re

from flask import Flask, request, render_template
# from sentence_transformers import SentenceTransformer
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
# model = SentenceTransformer(model_save_path)
model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

df = pd.read_csv("../data/corona_data.csv")

# Use English embedding instead of Swedish
df['question'] = df['question'].apply(lambda x: translator.translate(x, dest='en').text)
embedding_mat = model(list(df.question.values)).numpy()
# embedding_mat = model.encode(list(df.question.values))

UPDATED_RE = re.compile(r"(\s\w+: (?P<updated>\d\d\d\d-\d\d-\d\d\s\d\d:\d\d))")
LINK_RE = re.compile(r"(\s\[(?P<label>.*?)\]\((?P<url>.+?)\))")

LANGUAGES = ["ar", "en", "es", "fr", "de", "nl", "zh-cn", "ja", "ko", "pl", "pt", "th", "tr", "ru"]


def prepare_source_link(text, links, lang):
    source_link = links.pop(-1)
    text = text.replace(source_link["id"], "")

    source_link.pop("id")
    url = source_link["url"]
    source_link["url"] = f"?exp={url.rsplit('?exp=', 1)[1]}"
    source_link["label"] = translate(source_link["label"], lang)

    return text, source_link


def get_and_replace_links(text):
    link_matches = LINK_RE.findall(text)
    links = []

    for i, link in enumerate(link_matches):
        link_id = f"{i}" * 5
        text = text.replace(f"{link[0]}", link_id)
        links.append({"id": link_id, "label": link[1], "url": link[2]})

    return text, links


def insert_links(text, links, lang):
    for link in links:
        t_link_label = translate(link["label"], lang).capitalize()
        link_rep = f'<a href="https://www.folkhalsomyndigheten.se' \
                   f'{link["url"].replace(" ", "")}">' \
                   f'{t_link_label}</a>'
        text = text.replace(link["id"], link_rep)

    return text


def get_and_replace_updated(text):
    updated_match = UPDATED_RE.search(text)
    updated = updated_match["updated"]
    text = text.replace(updated_match.group(0), "")
    return text, updated


def translate(text, lang):
    return translator.translate(text, dest=lang).text


def translate_answer(answer, lang):
    answer, updated = get_and_replace_updated(answer)
    answer, links = get_and_replace_links(answer)
    answer, source_link = prepare_source_link(answer, links, lang)
    t_answer = translate(answer, lang)
    t_answer = insert_links(t_answer, links, lang)
    return t_answer, updated, source_link

@app.route("/")
def my_form():
    return render_template("hello.html")


@app.route("/", methods=["POST"])
def query_check():
    # query = model.encode([request.form["text"]])
    lang = request.form["lang"]
    if lang is None or lang not in LANGUAGES:
        lang = "en"

    query_text = request.form["text"]
    query = model([query_text])
    query_mat = np.full((len(df), np.array(embedding_mat).shape[1]), query)

    cosim_matrix = 1 - distance.cdist(
        query_mat, embedding_mat, "cosine").diagonal()

    translations = []

    picked_items = cosim_matrix.argsort()[-5:][::-1]
    answers = list(df.reset_index().iloc[picked_items].answer.values)
    questions = list(df.reset_index().iloc[picked_items].question.values)
    bs = np.round(cosim_matrix[picked_items], 2)

    for answer, question, b in zip(answers, questions, bs):
        translated_answer, updated, source_link = translate_answer(answer, lang)

        translated_question = translate(question, lang)

        result = {
            "answer": ". ".join(
                [scentence.capitalize() for scentence in
                 translated_answer.split(". ")]),
            "question": translated_question,
            "updated": updated,
            "source_link": source_link,
            "b": b,
        }

        translations.append(result)

    return render_template(
        "hello.html",
        translations=translations,
        query=query_text,
        language=lang,
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0')
