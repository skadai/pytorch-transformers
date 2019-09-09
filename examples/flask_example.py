# coding=utf-8
# Created by Meteorix at 2019/7/30

from flask import Flask, request, jsonify
from service_streamer import ThreadedStreamer
# from bert_model import TextInfillingModel as Model
from ecom_model import EcomSentiModel as Model


app = Flask(__name__)

# model_path = '/data/projects/bert_pytorch/skincare_op_out/all_in_one/checkpoint-4200'
# model = Model(model_path=model_path, target_device='cuda:0')
# streamer = ThreadedStreamer(model.predict, batch_size=32, max_latency=0.1)

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
    # print(outputs)
    return jsonify(outputs)


if __name__ == "__main__":
    model_path = '/data/projects/bert_pytorch/skincare_op_out/all_in_one/checkpoint-4200'
    model = Model(model_path=model_path, target_device='cuda:0')
    # start child thread as worker
    streamer = ThreadedStreamer(model.predict, batch_size=8, max_latency=0.01)

    # spawn child process as worker
    # streamer = Streamer(model.predict, batch_size=64, max_latency=0.1)

    app.run(host='0.0.0.0', port=5005, debug=True)
