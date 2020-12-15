from flask import Flask, render_template
from predict import multi_models, sentiment_scores
from Bert_load import bert_load


app = Flask(__name__)
@app.route('/')
def home():
    return render_template("web_ui.html", title='HOME')

@app.route('/search/<review>')
def search(review):
    review = str(review)
    #Simple RNN and LSTM-1D
    y_pred = main_process(review)
    #Vader 
    sentiment_dict = sentiment_scores(review)
    #Bert fine-tuning
    output_dir = "./model_save/"
    bert_test = bert_load(output_dir)
    label=bert_test.get_predict(review)
    if len(y_pred)==0:
        return {"state": "Empty result"}, 200
    else:
        json_return = {"state": "Ok", 
                        "rnn": { "key":["SimpleRNN","LSTM-1D"],
                                "result": y_pred,
                                "length": len(y_pred)
                                },
                        "vader": {"result": sentiment_dict},
                        "bert": int(label)
                                  }
        return json_return, 200

def main_process(review):
    tnz_path = 'tnz_raw.pickle'
    weights_path = ['best_model02.hdf5','best_model21.hdf5']
    mode = ["Raw","Raw"]
    y_pred=[]
    for i in range(2):
        mm = multi_models(tnz_path, weights_path[i], mode[i])
        y_pred.append(str(round(mm.postive_score(review),3)))
    print("Simple and LSTM-1D y_pred: ", y_pred)
    return y_pred

if __name__=="__main__":
    print("Hello world")
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=False)