# coding=utf-8
from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer
from ecom_model import EcomSentiModel as Model


app = Flask(__name__)


model = None
streamer = None


@app.route("/naive", methods=["POST"])
def naive_predict():
    form = request.get_json()
    inputs = form.get("texts")
    outputs = model.predict(inputs)
    return jsonify(outputs)


@app.route("/stream", methods=["POST"])
def stream_predict():
    form = request.get_json()
    inputs = form.get("texts")
    outputs = streamer.predict(inputs)
    return jsonify(outputs)


if __name__ == "__main__":
    model_path = '/data/projects/bert_pytorch/skincare_patch_out/all_in_one/checkpoint-4200'
    model = Model(model_path=model_path, target_device='cuda:0', subtype_dict='skincare')

    # start child thread as worker
    streamer = ThreadedStreamer(model.predict, batch_size=8, max_latency=0.01)

    # spawn child process as worker
    # streamer = Streamer(model.predict, batch_size=64, max_latency=0.1)

    app.run(host='0.0.0.0', port=5005, debug=True)
