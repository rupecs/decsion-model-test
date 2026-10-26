"""Loopback-only inference service and a local dashboard proxy."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_BODY = 32768


def reply(handler, status: int, value: dict) -> None:
    data=json.dumps(value,allow_nan=False).encode()
    handler.send_response(status)
    handler.send_header('Content-Type','application/json')
    handler.send_header('Cache-Control','no-store')
    handler.send_header('Content-Length',str(len(data)))
    handler.end_headers();handler.wfile.write(data)


def read_input(handler) -> dict:
    if handler.headers.get('Content-Type','').split(';')[0] != 'application/json':
        raise ValueError('Send JSON input.')
    size=int(handler.headers.get('Content-Length','0'))
    if not 0 < size <= MAX_BODY:
        raise ValueError('Input must be between 1 and 32768 bytes.')
    value=json.loads(handler.rfile.read(size))
    if not isinstance(value,dict) or set(value)!={'context','response'}:
        raise ValueError('Supply customer context and agent response.')
    if any(not isinstance(value[k],str) or not value[k].strip() for k in value):
        raise ValueError('Both input fields are required.')
    return value


def predictor_handler(predictor):
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass  # Never log submitted text.

        def do_GET(self):
            if self.path!='/health':
                return reply(self,404,{'error':'Not found'})
            return reply(self,200,{'ready':True,'model_version':predictor.config['project']['model_version'],
                'questions':predictor.config['dataset'].get('questions',{})})

        def do_POST(self):
            if self.path!='/predict':
                return reply(self,404,{'error':'Not found'})
            try:
                value=read_input(self)
            except (ValueError,TypeError):
                return reply(self,400,{'error':'Both fields must contain text; maximum request size is 32 KB.'})
            try:
                start=time.perf_counter()
                with lock:
                    prediction=predictor.predict(value['context'],value['response'])
                return reply(self,200,{'prediction':prediction,'latency_seconds':time.perf_counter()-start})
            except ValueError:
                return reply(self,422,{'error':'The model could not encode this input. Check the conversation and try shorter text.'})
            except Exception as exc:
                logging.error('Inference failed (%s)',type(exc).__name__)
                return reply(self,500,{'error':'Inference failed. Check GPU availability and service status.'})
    return Handler


def dashboard_handler(directory: Path, endpoint: str):
    target=urllib.parse.urlparse(endpoint)
    if target.scheme!='http' or target.hostname not in {'127.0.0.1','localhost'} or target.username or target.password or target.path not in {'','/'} or target.query or target.fragment:
        raise ValueError('Inference endpoint must be a loopback HTTP origin, normally an SSH tunnel.')
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,directory=str(directory),**kwargs)

        def log_message(self,*args):
            pass

        def proxy(self,path: str,body: bytes | None = None):
            try:
                request=urllib.request.Request(endpoint.rstrip('/')+path,data=body,
                    headers={'Content-Type':'application/json'} if body else {})
                with urllib.request.urlopen(request,timeout=120 if body else 3) as response:
                    return reply(self,response.status,json.load(response))
            except urllib.error.HTTPError as exc:
                try: value=json.load(exc)
                except ValueError: value={'error':'Inference service returned an error.'}
                return reply(self,exc.code,value)
            except (OSError,ValueError):
                return reply(self,503,{'ready':False,'error':'Inference is not connected yet. It becomes available after training and checkpoint verification.'})

        def do_GET(self):
            if self.path=='/api/inference/status':
                return self.proxy('/health')
            path=urllib.parse.urlparse(self.path).path
            if path not in {'/','/index.html','/live.json','/training.html'}:
                return reply(self,404,{'error':'Not found'})
            return super().do_GET()

        def do_POST(self):
            if self.path!='/api/predict':
                return reply(self,404,{'error':'Not found'})
            origin=self.headers.get('Origin')
            port=self.server.server_address[1]
            if origin and origin not in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}:
                return reply(self,403,{'error':'Use the local dashboard to submit input.'})
            try: body=json.dumps(read_input(self)).encode()
            except (ValueError,TypeError):
                return reply(self,400,{'error':'Enter customer context and agent response, within 32 KB.'})
            return self.proxy('/predict',body)
    return Handler


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['dashboard','model'],required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--directory',type=Path)
    p.add_argument('--endpoint',default='http://127.0.0.1:8788')
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args(argv)
    if a.mode=='dashboard':
        if not a.directory:p.error('--directory is required')
        handler=dashboard_handler(a.directory.resolve(),a.endpoint)
    else:
        if not a.checkpoint:p.error('--checkpoint is required')
        from .cli import parser,command_config
        from .inference import Predictor
        args=parser().parse_args(['predict','--checkpoint',str(a.checkpoint)])
        predictor=Predictor(command_config(args),str(a.checkpoint))
        handler=predictor_handler(predictor)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler)
    print(f'{a.mode} ready on 127.0.0.1:{a.port}',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
