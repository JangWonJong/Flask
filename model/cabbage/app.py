import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask, render_template, request
from cabbage.cab2_model import Cabbage

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('cabbage.html')

@app.route("/cabbage", methods=["post"])    
def cabbage():
    avgTemp = request.form['avgTemp']
    minTemp = request.form['minTemp']
    maxTemp = request.form['maxTemp']
    rainFall = request.form['rainFall']
    c =Cabbage()
    result = c.load_model(avgTemp, minTemp, maxTemp, rainFall)
    render_params = {}
    render_params['result'] = result
    return render_template('cabbage.html', **render_params)
   
    

if __name__=='__main__':
    print(f'Started Server')
    app.run()