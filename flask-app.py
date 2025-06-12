from flask import Flask, redirect, url_for, request, render_template,jsonify
import json
import os
app = Flask(__name__)


@app.route("/")
def index():
    #mensaje="Hola desde Flask!"
    #return render_template("index.html", mensaje=mensaje)
    return render_template("index.html")

@app.route("/map")
def map():    
    return render_template("map.html")
@app.route('/api/properties')
def get_properties():
    with open('static/data/properties.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/test2')
def test():
    return render_template("test2.html")

@app.route('/basic')
def basic():
    return render_template("basic.html")

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


if __name__ == '__main__':
    if not os.path.exists('static/data/properties.json'):
        from preprocess import preprocess_txt_to_json
        preprocess_txt_to_json('depaAlqUrbaniaConsolidado.txt', 'static/data/properties.json')
    app.run(debug=True)
