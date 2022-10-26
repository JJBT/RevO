import os
from flask import Flask, render_template, Response, send_file, jsonify, request
import urllib
import threading


PORT = 7860
LOCALHOST_NAME = '127.0.0.1'
TEMPLATE_PATH = 'templates/'
STATIC_PATH = 'static/'

app = Flask(__name__,
            template_folder=TEMPLATE_PATH,
            static_folder=STATIC_PATH
            )


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html',
                           config=app.interface.config,
                           # input_interfaces=[interface[0]
                           #                   for interface in app.interface.config["input_interfaces"]],
                           # output_interfaces=[interface[0]
                           #                    for interface in app.interface.config["output_interfaces"]],
                           )


@app.route('/config/', methods=['GET'])
def config():
    return jsonify(app.interface.config)


@app.route('/predict/', methods=['POST'])
def predict():
    input = request.json['data']
    output = app.interface.run_prediction(input)
    return jsonify({
        'data': output
    })


def start_server(interface):
    app.interface = interface
    app_kwargs = {'port': PORT, 'host': LOCALHOST_NAME}
    thread = threading.Thread(target=app.run, kwargs=app_kwargs, daemon=True)
    thread.start()
    return {
        'app': app,
        'thread': thread,
        'host': LOCALHOST_NAME,
        'port': PORT
    }


def close_server(process):
    process.terminate()
    process.join()


def url_request(url):
    try:
        req = urllib.request.Request(
            url=url, headers={"content-type": "application/json"}
        )
        return urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        raise RuntimeError(str(e))
