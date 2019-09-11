# -*- coding: utf-8 -*-

# @File    : gunicorn_conf.py
# @Date    : 2019-09-06
# @Author  : skym
import logging

from gevent import monkey; monkey.patch_all()
from service_streamer import ThreadedStreamer

from ecom_model import EcomSentiModel as Model

LOG_PATH = '/data/logs/ecom_senti/'
LOG_FILE = 'ecom_senti.log'
PID_FILE = 'ecom_senti.pid'


def post_fork(server, worker):
    import flask_example
    model_path = '/data/projects/bert_pytorch/skincare_patch_out/all_in_one_ground/checkpoint-4200'
    model = Model(model_path=model_path, target_device='cpu', subtype_dict='skincare')
    flask_example.streamer = ThreadedStreamer(model.predict, batch_size=32, max_latency=0.2)
    flask_example.model = model


bind = '%s:%s' % ('0.0.0.0', 5005)
environment = CUDA_VISIBLE_DEVICES="0"
preload_app = True
workers = 1
worker_class = 'gunicorn.workers.ggevent.GeventWorker'
worker_connections = 100
timeout = 120
deamon = False
debug = False
log_level = logging.INFO
pidfile = '%s/%s' % (LOG_PATH, PID_FILE)
logfile = '%s/%s' % (LOG_PATH, LOG_FILE)
