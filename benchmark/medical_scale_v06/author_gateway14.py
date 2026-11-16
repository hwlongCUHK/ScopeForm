"""Loopback-only durable journal for finite data-authoring requests.

An SSH disconnect cannot cause a second generation: callers recover the same
job ID. This service never imports an evaluator or any task standard labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

LOCK = threading.RLock()


def body_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def job_id(request):
    return hashlib.sha256(body_bytes(request)).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending.' + str(threading.get_ident()))
    with temporary.open('wb') as f:
        f.write(body_bytes(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, root, backend_port, workers=4):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend_port = backend_port
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.active = set()

    def folder(self, key):
        if not re.fullmatch('[0-9a-f]{64}', key):
            raise ValueError('Unsafe job ID')
        return self.root / key

    def get(self, key):
        folder = self.folder(key)
        if (folder / 'response.private.json').exists():
            return {'job_id': key, 'state': 'done',
                    'response': json.loads((folder / 'response.private.json').read_text(encoding='utf-8'))}
        if (folder / 'failure.private.json').exists():
            return {'job_id': key, 'state': 'failed',
                    'failure': json.loads((folder / 'failure.private.json').read_text(encoding='utf-8'))}
        if not (folder / 'request.private.json').exists():
            return {'job_id': key, 'state': 'absent'}
        if (folder / 'dispatched.json').exists() and key not in self.active:
            return {'job_id': key, 'state': 'ambiguous_inflight_no_automatic_regeneration'}
        return {'job_id': key, 'state': 'running' if (folder / 'dispatched.json').exists() else 'queued'}

    def register(self, key, request):
        if job_id(request) != key:
            raise ValueError('Request hash differs from job ID')
        if request.get('model') != 'Qwen3.8-27B' or request.get('stream') is not False:
            raise ValueError('Only the predeclared local author model')
        if not 1 <= request.get('max_tokens', 0) <= 12288 or 'tools' in request:
            raise ValueError('Unexpected request profile')
        folder = self.folder(key)
        with LOCK:
            if (folder / 'request.private.json').exists():
                prior = json.loads((folder / 'request.private.json').read_text(encoding='utf-8'))
                if prior != request:
                    raise ValueError('Do not replace a journaled request')
            else:
                folder.mkdir(parents=True, exist_ok=True)
                atomic(folder / 'request.private.json', request)
            state = self.get(key)
            if state['state'] == 'queued' and key not in self.active:
                self.active.add(key)
                self.executor.submit(self.run, key, request)
            return self.get(key)

    def run(self, key, request):
        folder = self.folder(key)
        try:
            atomic(folder / 'dispatched.json', {'time': time.time(), 'request_sha256': key})
            req = urllib.request.Request('http://127.0.0.1:' + str(self.backend_port) + '/v1/chat/completions',
                data=body_bytes(request), headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=1800) as response:
                    raw = json.load(response)
                atomic(folder / 'response.private.json', raw)
            except urllib.error.HTTPError as error:
                content = error.read()
                atomic(folder / 'failure.private.json', {'kind': 'http_error', 'status': error.code,
                    'private_detail': content.decode('utf-8', errors='replace')[:20000]})
            except Exception as error:
                atomic(folder / 'failure.private.json', {'kind': 'backend_transport_ambiguous_not_retried',
                    'error_type': type(error).__name__})
        finally:
            with LOCK:
                self.active.discard(key)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, value, code=200):
        raw = body_bytes(value)
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/health':
            return self.reply({'ok': True, 'kind': 'authoring_journal_not_solver'})
        try:
            if not self.path.startswith('/jobs/'):
                raise ValueError('Unknown route')
            return self.reply(self.server.journal.get(self.path[len('/jobs/'):]))
        except ValueError as error:
            return self.reply({'error': str(error)}, 400)

    def do_POST(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if self.path != '/jobs' or not 0 < size <= 4*1024*1024:
                raise ValueError('Invalid route or body size')
            payload = json.loads(self.rfile.read(size))
            if set(payload) != {'job_id', 'request'}:
                raise ValueError('Unexpected job envelope')
            return self.reply(self.server.journal.register(payload['job_id'], payload['request']))
        except (ValueError, KeyError, TypeError) as error:
            return self.reply({'error': str(error)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18641)
    parser.add_argument('--backend-port', type=int, default=18640)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    allowed = Path('/data/zhanchen/cpm_medical06_20261020/medical_scale_v06/construction/confirmation14_author_journal_01')
    if os.name != 'posix' or args.journal.resolve() != allowed or args.port != 18641 or args.backend_port != 18640:
        raise ValueError('Only the named finite journal and loopback profile')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.journal = Journal(args.journal, args.backend_port)
    server.serve_forever()


if __name__ == '__main__':
    main()
