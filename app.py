# import necessary libraries
import numpy as np
#import pickle as pkl
import joblib
import os
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect,
    url_for)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

# from flask_sqlalchemy import SQLAlchemy
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '') or "sqlite:///db.sqlite"

# # Remove tracking modifications
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

model = joblib.load('decisionTree_model.sav', mmap_mode=None)

# create route that renders index.html template
@app.route("/")
def index ():
    return render_template("index.html")

@app.route('/index.html', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/research_methodology.html')
def reasearh():

    return render_template("research_methodology.html")

@app.route("/dataset_observation.html")
def obsserv():
    return render_template("dataset_observation.html")

@app.route("/dataset_analysis.html")
def data_analysis():
    return render_template("dataset_analysis.html")

@app.route("/model_presentation.html")
def dataset():
    return render_template("model_presentation.html")

# @app.route("/challenges")
# def dataset():
#     return render_template("challenges.html")

# @app.route("/Q&A")
# def dataset():
#     return render_template("Q&A.html")

@app.route("/model_prediction.html")
def redirectPredict():
    return render_template('/model_prediction.html')

@app.route("/send", methods=["GET", "POST"])
def send():

    data = []
    if request.method == "POST":
    
        vMatch = request.form["match"]
      
        vLike = request.form["like"]
        vAttr = request.form["attr"]
        vDec_o = request.form["dec_o"]
        vProb = request.form["prob"]
        vSinc = request.form["sinc"]
        vAttr_2_1 = request.form["attr_2_1"]
        vAttr_1 = request.form["attr_1"]
 
        
        data.append(vMatch)
        data.append(vLike)
        data.append(vAttr)
        data.append(vDec_o)
        data.append(vProb)
        data.append(vSinc)
        data.append(vAttr_2_1)
        data.append(vAttr_1)
       
    
    #print(data)
    predict = model.predict(np.array(data).reshape(1, -1))
    #print(predict)
    
    result = predict[0]
    
    if result == 1:
        result ='Yes'
    else:
        result ='No'
    #return jsonify(result)
    #return str(result)
    return render_template('model_prediction.html', predict_text = 'Your Decision: {}'.format(str(result)))

 

if __name__ == "__main__":
    app.run(debug=True)
