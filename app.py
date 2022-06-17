from flask import Flask
import os
import sys

from model.calc_model import CalcModel
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/")
def calc():
    num1 = request.form['nun1']
    num2 = request.form['nun2']
    opcode = request.form['opcode']
    calc = CalcModel()
    result = calc.calc(num1, num2, opcode)
    render_params = {}
    render_params['result'] = result
    return render_template('index.html', **render_params)

if __name__=='main':
    print(f'Started Server')
    app.run()