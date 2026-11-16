"""Gold-free paired layouts, revisioned evidence and authorization via HTTP.

This loopback simulator is not a deployed hospital or an independent medical
validation. Identical clinical text is reorganized, never rewritten.
"""
import copy
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import urllib.request
import uuid

import confirmation14_workflow as h
from pilot_runtime import fingerprint
from run_generation import NoRedirect

LAYOUTS=('centralized_current_materials','distributed_versioned_services')
EVENTS=('critical_new_version','authorization_withdrawal','neutral_version_change')
PARENT_RESTORE=h.restore


def controller(public,layout,revision,withdraw=False):
    if layout not in LAYOUTS:raise ValueError('Undeclared paired layout')
    value=copy.deepcopy(public);mapping={b['anchor_id']:b['anchor_id']+'__web_v'+str(revision) for d in value['documents'] for b in d['blocks']}
    for d in value['documents']:
        for b in d['blocks']:b['anchor_id']=mapping[b['anchor_id']]
        d['document_id']='WEB_SERVICE__'+d['document_id']
    if layout==LAYOUTS[0]:
        value['documents']=[{'document_id':'WEB_CENTRAL__'+value['source_id'],
            'title':'Current authorized materials, single consolidated service',
            'blocks':[b for d in value['documents'] for b in d['blocks']]}]
    for claim in value['claims']:
        if 'reference_locator' in claim:
            locator = claim['reference_locator']
            locator['anchor_id'] = mapping[locator['anchor_id']]
    value['visibility']={stage:[mapping[a] for a in ids] for stage,ids in value['visibility'].items()}
    value.update(environment='E3',key_revokes_write=withdraw,
        task_version='confirmation15_web_v1',task_id=value['source_id']+'__web15_'+layout)
    return value,mapping


def actor(original,layout):
    initial,_=controller(original['initial'],layout,0)
    critical=next(e['controller'] for e in original['events'] if e['event_id']=='key_mixed')
    updated,_=controller(critical,layout,1)
    auth,_=controller(original['initial'],layout,0,True)
    neutral,_=controller(original['initial'],layout,1)
    return {'source_id':original['source_id'],'initial':initial,'events':[
        {'event_id':EVENTS[0],'controller':updated},{'event_id':EVENTS[1],'controller':auth},
        {'event_id':EVENTS[2],'controller':neutral}]}


def target_items(items,mapping):
    result=copy.deepcopy(items)
    for g in result:
        for field in ('support_anchor_ids','required_visible_anchor_ids'):
            if field in g:g[field]=[mapping.get(x,x) for x in g[field]]
        g['sufficient_evidence_groups']=[[mapping[x] for x in group] for group in g['sufficient_evidence_groups']]
    return result


class WebCore(h.World):
    def __init__(self,public,layout=LAYOUTS[0],revision=0):
        super().__init__(public);self.web_layout=layout;self.web_revision=revision;self.web_http=True

    def initial_input(self):
        result=super().initial_input()
        result['web_service_contract']={'layout':self.web_layout,'current_evidence_revision':self.web_revision,
            'clinical_materials':'Identical original text and explicit synthetic current states; source placement and evidence versions vary.',
            'read_version_rule':'Only current acquired version-scoped anchors can substantiate this phase. Prior summary may contain expired citations.',
            'write_authorization':self.permission(),'transport':'Public tools are loopback HTTP service operations, not external browsing.'}
        return result

    def call(self,name,args):
        if name in ('save_summary','submit_answer') and isinstance(args,dict) and self.shape(args.get('items')):
            current={b['anchor_id'] for d in self.documents() for b in d['blocks']}
            if any(ref not in current for i in args['items'] for ref in i['evidence_ids']):
                result={'error':'stale_or_unknown_evidence_version','current_revision':self.web_revision,'correctness_disclosed':False}
                self.trace.append({'tool':name,'arguments':copy.deepcopy(args),'result':result});return result
        return super().call(name,args)

    def _call(self,name,args):
        result=super()._call(name,args)
        if name=='list_sources' and 'sources' in result:
            for d in result['sources']:d.update(service_revision=self.web_revision,current_only=True)
        return result


def restore_core(data):
    world=PARENT_RESTORE(data);world.__class__=WebCore
    return world


class ToolService:
    def __init__(self,port=0):
        sessions={};guard=threading.Lock();self.sessions=sessions
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<32*1024**2:raise ValueError('Invalid request length')
                    body=json.loads(self.rfile.read(n))
                    with guard:
                        if self.path=='/create':
                            token=uuid.uuid4().hex;sessions[token]=restore_core(body['world']);result={'session_id':token}
                        elif self.path=='/invoke':
                            w=sessions[body['session_id']];action=body['action']
                            value=w.call(action['name'],action['arguments'])
                            result={'result':value,'world':h.prior.snapshot(w)}
                        elif self.path=='/close':sessions.pop(body['session_id'],None);result={'closed':True}
                        else:raise ValueError('Unknown service operation')
                    raw=json.dumps(result,ensure_ascii=False).encode();self.send_response(200)
                except Exception as error:
                    raw=json.dumps({'error':type(error).__name__}).encode();self.send_response(400)
                self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        self.server=ThreadingHTTPServer(('127.0.0.1',port),Handler);self.server.daemon_threads=True
        self.port=self.server.server_port;self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)

    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join(timeout=5)


class WebWorld(WebCore):
    def __init__(self,public,layout=LAYOUTS[0],revision=0,port=0):
        super().__init__(public,layout,revision);self.web_http_port=port;self.web_session=None

    def rpc(self,path,body):
        req=urllib.request.Request('http://127.0.0.1:'+str(self.web_http_port)+path,
            data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=60) as response:return json.load(response)

    def call(self,name,args):
        if not self.web_session:self.web_session=self.rpc('/create',{'world':h.prior.snapshot(self)})['session_id']
        reply=self.rpc('/invoke',{'session_id':self.web_session,'action':{'name':name,'arguments':args}})
        token=self.web_session;port=self.web_http_port
        self.__dict__.update(restore_core(reply['world']).__dict__);self.web_session=token;self.web_http_port=port
        return reply['result']

    def close(self):
        if self.web_session:self.rpc('/close',{'session_id':self.web_session})
        # Keep immutable terminal checkpoint identity; do not rewrite session ID.


def restore_proxy(data):
    world=restore_core(data);world.__class__=WebWorld;world.web_session=None
    return world


def checkpoint_hash(world_or_snapshot):
    value=h.prior.snapshot(world_or_snapshot) if not isinstance(world_or_snapshot,dict) else copy.deepcopy(world_or_snapshot)
    for key in ('web_session','web_http_port'):value.pop(key,None)
    return fingerprint(value)


def branch(base,target,event,method,port):
    world=h.branch(base,target,event,method);world.__class__=WebWorld
    world.web_layout=base.web_layout;world.web_revision=0 if event=='authorization_withdrawal' else 1
    world.web_http_port=port;world.web_session=None;world.web_http=True
    world.event_notice['kind']={'critical_new_version':'medical_facts_and_evidence_revision_changed',
        'authorization_withdrawal':'same_medical_facts_write_authorization_withdrawn',
        'neutral_version_change':'same_medical_facts_new_evidence_revision'}[event]
    if fingerprint(world.state)!=fingerprint(base.state):raise ValueError('Actual persistence changed')
    return world
