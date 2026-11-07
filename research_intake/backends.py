"""Model backends behind one interface. Local only; never a cloud endpoint.

  * OllamaBackend: local Ollama on 127.0.0.1, as in the procurement pilot. Refuses
    to run a model that is not installed (no silent substitution) and records the
    model digest and runtime version in the job identity.
  * StubBackend: deterministic keyword matcher for testing the plumbing without a
    GPU. Its output is labelled backend=stub everywhere and is never a model result.

Swap models by editing config/routes.json, not code. When the selected parser
routes (PaddleOCR-VL tables, Nemotron Parse pages) are provisioned on-prem, add
an adapter here with the same generate() contract.
"""
import json
from pathlib import Path
import re
import urllib.request

from .store import canonical, digest

ROUTES = Path(__file__).resolve().parent / 'config' / 'routes.json'


class OllamaBackend:
    base = 'http://127.0.0.1:11434'

    def __init__(self, route):
        self.route = route
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _api(self, path, payload=None):
        request = urllib.request.Request(self.base + path, headers={'Content-Type': 'application/json'},
                                         data=None if payload is None else canonical(payload).encode())
        with self.http.open(request, timeout=900) as response:
            return json.load(response)

    def identity(self):
        tags = {m['name']: m for m in self._api('/api/tags')['models']}
        if self.route['model'] not in tags:
            raise ValueError('Requested model is not installed; refusing to substitute another model')
        return {'backend': 'ollama', 'model': self.route['model'], 'digest': tags[self.route['model']]['digest'],
                'runtime': self._api('/api/version')['version'], 'context': self.route.get('context', 16384),
                'output_tokens': self.route.get('output_tokens', 3000), 'temperature': 0}

    def busy(self):
        return bool(self._api('/api/ps')['models'])

    def generate(self, identity, prompt, schema, images=None):
        message = {'role': 'user', 'content': prompt}
        if images:
            message['images'] = images
        raw = self._api('/api/chat', {'model': identity['model'], 'stream': False, 'format': schema,
                                      'keep_alive': '10m', 'messages': [message],
                                      'options': {'temperature': 0, 'num_ctx': identity['context'],
                                                  'num_predict': identity['output_tokens']}})
        return {'content': raw.get('message', {}).get('content', ''), 'done_reason': raw.get('done_reason'), 'raw': raw}

    def unload(self, identity):
        self._api('/api/generate', {'model': identity['model'], 'keep_alive': 0})


class StubBackend:
    """Keyword stub. Exercises scheduling, validation, consolidation and export only."""

    def __init__(self, route=None):
        self.route = route or {}

    def identity(self):
        return {'backend': 'stub', 'model': 'stub-keyword-v1', 'digest': digest(Path(__file__).read_bytes()),
                'runtime': 'none', 'context': 0, 'output_tokens': 0, 'temperature': 0}

    def busy(self):
        return False

    def unload(self, identity):
        pass

    @staticmethod
    def _hits(terms, text):
        return any(re.search(r'(?<![A-Za-z])' + re.escape(t) + r'(?![A-Za-z])', text, re.I) for t in terms)

    def generate(self, identity, prompt, schema, images=None):
        if images:
            # The stub cannot read images; report it rather than inventing text.
            return {'content': json.dumps({'text': ''}), 'done_reason': 'stop', 'raw': {'stub': 'cannot_read_images'}}
        request = self._parse(prompt)
        out = {}
        for f in request['fields']:
            hits_by = {}
            for key, terms in request['options'].get(f['id'], {}).items():
                hits_by[key] = [e['id'] for e in request['excerpts'] if self._hits(terms, e['text'])][:3]
            keyword_hits = [e['id'] for e in request['excerpts'] if self._hits(request['keywords'].get(f['id'], []), e['text'])][:3]
            item = {'outcome': 'not_found_in_excerpts', 'evidence': []}
            if f['type'] == 'enum':
                found = [(k, v) for k, v in hits_by.items() if v]
                item['value'] = None
                if found:
                    item.update(outcome='found', value=found[0][0], evidence=found[0][1])
            elif f['type'] == 'multi_enum':
                values = [{'value': k, 'evidence': v} for k, v in sorted(hits_by.items()) if v]
                item['values'] = values
                if values:
                    item['outcome'] = 'found'
            elif f['type'] == 'labels':
                text = {e['id']: e['text'] for e in request['excerpts']}
                item['labels'] = [{'label': 'STUB: ' + ' '.join(text[i].split()[:6]), 'evidence': [i]} for i in keyword_hits]
                if keyword_hits:
                    item['outcome'] = 'found'
            elif keyword_hits:
                item.update(outcome='found', evidence=keyword_hits)
            out[f['id']] = item
        return {'content': json.dumps({'fields': out}), 'done_reason': 'stop', 'raw': {'stub': True}}

    @staticmethod
    def _parse(prompt):
        # The stub reads the same prompt the real model gets, plus the schema's hint terms.
        from .evidence import load_schema, options
        fields = json.loads(prompt.split('\nFIELDS: ', 1)[1].split('\nEXCERPTS: ', 1)[0])
        excerpts = json.loads(prompt.split('\nEXCERPTS: ', 1)[1])
        schema = load_schema()
        defs = {f['id']: f for f in schema['fields']}
        return {'fields': fields, 'excerpts': excerpts,
                'options': {f['id']: options(schema, defs[f['id']]) for f in fields},
                'keywords': {f['id']: defs[f['id']].get('keywords', []) for f in fields}}


def load_routes():
    return json.loads(ROUTES.read_text(encoding='utf8'))


def backend_for(stage, override=None, model=None):
    route = dict(load_routes()['stages'][stage])
    if model:
        route['model'] = model
    kind = override or route['backend']
    return StubBackend(route) if kind == 'stub' else OllamaBackend(route)
