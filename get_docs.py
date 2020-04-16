import os, re, ast, _ast
import pandas as pd
import numpy as np
import nltk
import spacy


def get_module_paths(MODULES_PATH):
    """Extract all absolute module paths from Sympathy directory"""
    return [
        MODULES_PATH + "/" + f + "/" + fu
        for f in os.listdir(MODULES_PATH)
        for fu in os.listdir(MODULES_PATH + "/" + f)
        if "node" in fu
    ]


def get_variables(node):
    """Extract all variable description fields from Sympathy
    modules"""
    variables = []
    if hasattr(node, "body"):
        for subnode in node.body:
            if isinstance(subnode, (ast.FunctionDef, ast.ClassDef)):
                if (
                    subnode.body
                    and isinstance(subnode.body[0], ast.Expr)
                    and isinstance(subnode.body[0].value, ast.Str)
                ):
                    variables.append(subnode.body[0].value.s)
                else:
                    variables.extend(get_variables(subnode))
            else:
                variables.extend(get_variables(subnode))
    elif isinstance(node, _ast.Assign):
        for name in node.targets:
            if isinstance(name, _ast.Name):
                if name.id == "description":
                    variables.append(node.value.s)
    return variables


def get_docs(node):
    """Extract all docstrings and variable descriptions from Sympathy
    modules"""
    if get_variables(node) == set() or get_variables(node) == set({""}):
        doc = set()
        doc.add(str(ast.get_docstring(node)))
        return doc
    else:
        return get_variables(node)


def clean_docstring(doc_string):
    """Remove all unwanted string components
    from Sympathy docstring"""
    doc_string = " ".join(list(doc_string))  # Set object to string
    sentence_string = nltk.sent_tokenize(doc_string)[:1][0]  # Limit to first sentence
    clean_string = re.sub("[^A-Za-z0-9]+", " ", sentence_string).lower().replace("ref", "").split()
    nlp_string = nlp(" ".join(clean_string))
    final_string_list = []
    for i in nlp_string:
        if i.pos_ == 'NOUN' and i.text not in ['data']:
            final_string_list.append(lemmatizer(i.text, i.pos_)[0])
        else:
            final_string_list.append(i.text)
    return " ".join(final_string_list)


if __name__ == "__main__":

    nlp = spacy.load("en_core_web_md")
    lemmatizer = spacy.lemmatizer.Lemmatizer(spacy.lang.en.LEMMA_INDEX,
                                             spacy.lang.en.LEMMA_EXC, spacy.lang.en.LEMMA_RULES)
    stoplist = nltk.corpus.stopwords.words("english")
    MODULES_PATH = "/Users/juriegermishuys/anaconda3/envs/combine-demo/lib/python3.6/site-packages/" \
                   "sympathy_app/Library/Library/sympathy"
    nodes = get_module_paths(MODULES_PATH)
    node_desc = [
        clean_docstring(get_docs(ast.parse(open(i, encoding="utf-8").read())))
        for i in nodes
    ]
    df = pd.DataFrame(
        np.column_stack([nodes, node_desc]), columns=["node_path", "node_doc"]
    )
    df["base_path"] = df["node_path"].apply(lambda x: os.path.basename(x))
    df.to_csv("data/data.csv")
