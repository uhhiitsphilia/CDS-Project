from flask import Flask, render_template
from predict import multi_models
import json


app = Flask(__name__)
@app.route('/')
def home():
    return render_template("web_ui.html", title='HOME')


# @app.route('/search/<int:questionID>')
# def search(questionID):
#     connection = sqlite3.connect("links.db")
#     c = connection.cursor()
#     try:
#         c.execute(f"SELECT id, url, name, hardness FROM questions WHERE id=({questionID});")
#         result = c.fetchall()
#     except sqlite3.OperationalError:
#         return {"result":  "An error occurred"}, 200
#     if len(result) == 0:
#         return {"result": "Item not found"}, 200
#     else:
#         return {"result": result}, 200

@app.route('/search/<review>')
def search(review):
    review = str(review)
    y_pred = main_process(review)
    # print(y_pred)
    if len(y_pred)==0:
        return {"state": "Empty result"}, 200
    else:
        return {"state": "Ok", "key":["stem","lemm"],"result": y_pred,"length": len(y_pred)}, 200

def main_process(review):
    tnz_path = ['tnz_stem.pickle','tnz_lemm.pickle']
    weights_path = ['BiLSTM_stem.h5','BiLSTM_lemm.h5']
    mode = ["PorterStemmer","WordNetLemmatizer"]
    y_pred=[]
    for i in range(2):
        mm = multi_models(tnz_path[i], weights_path[i], mode[i])
        y_pred.append(float(mm.postive_score(review)))
    return y_pred

if __name__=="__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=False)

