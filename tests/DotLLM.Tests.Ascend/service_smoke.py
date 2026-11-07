#!/usr/bin/env python3
"""Actual CLI/OpenAI API acceptance on a single Ascend worker (Python is test-only)."""
import argparse
import concurrent.futures
import json
import os
import pty
import select
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dotnet', required=True)
    parser.add_argument('--cli', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [args.dotnet, args.cli]
    evidence = []
    def run(name, arguments, input=None):
        result = subprocess.run(command + arguments, input=input, text=True, capture_output=True, timeout=300)
        (args.output / (name + '.stdout')).write_text(result.stdout)
        (args.output / (name + '.stderr')).write_text(result.stderr)
        return result
    def record(name, **data):
        item = {'test': name, 'status': 'PASS', **data}
        evidence.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        (args.output / 'acceptance.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in evidence)+'\n')
    for index, model in enumerate(args.models):
        for repeat in range(3):
            result = run(f'run-{index}-{repeat}', ['run', model, '--device', 'ascend:0', '--prompt',
                         'Explain what a key value cache does in a transformer.', '--max-tokens', '32',
                         '--temp', '0', '--seed', '42', '--json'])
            assert result.returncode == 0, result.stderr
            response = json.loads(result.stdout)
            assert response['usage']['prompt_tokens'] > 0 and response['usage']['generated_tokens'] > 0, response
            assert response['memory']['weights_bytes'] > 0 and response['memory']['kv_cache_bytes'] > 0
            record('cli_run', model=Path(model).name, repeat=repeat, response=response)
    result = run('cpu-regression', ['run', args.models[0], '--device', 'cpu', '--threads', '4',
                 '--prompt', 'Hello world', '--max-tokens', '4', '--temp', '0', '--json'])
    assert result.returncode == 0, result.stderr
    record('cpu_regression', response=json.loads(result.stdout))
    for flags in [['--paged'], ['--gpu-layers', '0'], ['--cache-type-k', 'q8_0'], ['--speculative-model', 'missing']]:
        result = run('invalid-'+flags[0][2:], ['run', args.models[0], '--device', 'ascend:0', '--prompt', 'hello']+flags)
        assert result.returncode != 0 and 'unsupported' in (result.stdout+result.stderr).lower(), result
    record('invalid_flags', count=4)
    # A PTY exercises the actual interactive chat loop and requires both generation results.
    master, slave = pty.openpty()
    chat_process = subprocess.Popen(command+['chat', args.models[-1], '--device', 'ascend:0',
                    '--max-tokens', '8', '--temp', '0', '--verbose'], stdin=slave, stdout=slave,
                    stderr=slave, env={**os.environ, 'TERM':'xterm'})
    os.close(slave)
    transcript = b''
    stage = 0
    deadline = time.monotonic()+300
    try:
        while chat_process.poll() is None:
            if time.monotonic()>deadline: raise TimeoutError('Chat PTY timeout')
            if select.select([master], [], [], 1)[0]:
                try: data=os.read(master, 65536)
                except OSError: break
                transcript += data
                if b'\x1b[6n' in data: os.write(master,b'\x1b[1;1R')
                if stage == 0 and b'>>>' in transcript:
                    os.write(master,b'Hello\r'); stage=1
                elif stage == 1 and b'finish_reason=' in transcript and b'>>>' in transcript.split(b'finish_reason=')[-1]:
                    os.write(master,b'What did I just say?\r'); stage=2
                elif stage == 2 and transcript.count(b'finish_reason=')>=2 and b'>>>' in transcript.split(b'finish_reason=')[-1]:
                    os.write(master,b'/quit\r'); stage=3
        chat_process.wait(timeout=10)
    finally:
        if chat_process.poll() is None: chat_process.kill(); chat_process.wait()
        os.close(master)
        (args.output/'chat.pty').write_bytes(transcript)
    assert chat_process.returncode == 0 and transcript.count(b'finish_reason=') >= 2, transcript.decode(errors='replace')
    record('cli_chat', turns=2, terminal='PTY')
    base = 'http://127.0.0.1:18080'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, body=None):
        req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        return opener.open(req, timeout=300)
    def api(path, body=None):
        with request(path, body) as response:
            return json.load(response)
    log = (args.output/'server.log').open('w')
    server = subprocess.Popen(command+['serve', args.models[-1], '--device', 'ascend:0', '--host', '127.0.0.1',
                              '--port', '18080', '--no-browser', '--no-ui', '--warmup-iterations', '1'],
                              stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+180
        while True:
            if server.poll() is not None: raise RuntimeError('Server exited during startup')
            try:
                if api('/ready')['status'] == 'ready': break
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            if time.monotonic()>deadline: raise TimeoutError('Server readiness timeout')
            time.sleep(0.5)
        assert api('/health')['status']=='ok'
        assert len(api('/v1/models')['data']) == 1
        text='Hello 世界\nprint(1)'
        tokenized=api('/v1/tokenize', {'text': text})
        assert tokenized['count'] == len(tokenized['tokens']) > 0
        assert api('/v1/detokenize', {'tokens': tokenized['tokens']})['text'] == text
        record('server_health_tokenizer_warmup')
        completion={'model':'test', 'prompt':'The purpose of a key value cache is', 'max_tokens':16, 'temperature':0, 'seed':42}
        def validate(value):
            assert value['choices'] and value['choices'][0]['finish_reason'] in ('stop','length')
            usage=value['usage']; assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
            assert 0 <= usage['completion_tokens'] <= 16
            return value
        first=validate(api('/v1/completions', completion))
        second=validate(api('/v1/completions', completion))
        assert first['choices'][0]['text']==second['choices'][0]['text']
        record('completion_repeat', response=first)
        chat={'model':'test','messages':[{'role':'user','content':'Hello'}, {'role':'assistant','content':'Hello!'},
              {'role':'user','content':'What did I just say?'}], 'max_tokens':16,'temperature':0,'seed':42}
        record('chat_multiturn', response=validate(api('/v1/chat/completions', chat)))
        for path,body in [('/v1/completions', completion),('/v1/chat/completions',chat)]:
            with request(path, {**body,'stream':True}) as stream:
                raw=stream.read().decode()
            assert raw.rstrip().endswith('data: [DONE]'),raw
            chunks=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            assert chunks and any(c['choices'] and c['choices'][0].get('finish_reason') in ('stop','length') for c in chunks)
            (args.output/('chat.sse' if 'chat' in path else 'completion.sse')).write_text(raw)
            record('sse', endpoint=path, chunks=len(chunks), done=True)
        # Disconnect an active SSE response, then require a subsequent request to complete.
        stream=request('/v1/completions', {**completion,'stream':True,'max_tokens':512})
        while True:
            line=stream.readline()
            if not line: raise RuntimeError('SSE closed before cancellation')
            if line.startswith(b'data: {'): break
        stream.close()
        validate(api('/v1/completions', completion))
        record('disconnect_recovery')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(api, '/v1/completions', completion) for _ in range(2)]
            results=[validate(f.result()) for f in futures]
        assert all(x['choices'][0]['text']==first['choices'][0]['text'] for x in results)
        record('serialized_concurrency', clients=2, independent_state=True)
        try:
            api('/v1/completions', {**completion,'max_tokens':-1})
            raise AssertionError('Invalid request accepted')
        except urllib.error.HTTPError as error:
            assert error.code==400
        record('api_invalid_request')
        # A failed model load cleans up the attempted model; the documented bare state can recover.
        try:
            api('/v1/models/load', {'model':args.models[-1],'device':'ascend:0','cache_type_k':'q8_0'})
            raise AssertionError('Unsupported cache load accepted')
        except urllib.error.HTTPError as error:
            assert error.code==400
        loaded=api('/v1/models/load', {'model':args.models[-1],'device':'ascend:0','cache_type_k':'f32'})
        assert loaded['status']=='loaded'
        validate(api('/v1/completions', completion))
        record('model_reload_recovery')
    finally:
        server.terminate()
        try: server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        log.close()
    record('service_suite', checks=len(evidence))

if __name__=='__main__':
    main()
