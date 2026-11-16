"""Gold-free shared serial JSON-action harness for the prospective study.

No model-family-specific native function parser is used. Tools, intervention
semantics, decoding and action limits are shared across the three families.
This is a new harness study; do not merge it with older native-tool rankings.
"""
import copy
import json
import time
import urllib.error
import urllib.request

from run_generation import read, dump, NoRedirect
from pilot_runtime import fingerprint
from token_count import count_chat_tokens
from workflow11_runtime import World, POLICY, fork, export_observations
from workflow12 import FactorialWorld, fork12
import run_workflow11 as prior

METHODS=('labels_only','index_only','labels_guard','evidence_constraint','fresh')


def restore(data):
    world=prior.restore(data)
    if 'factorial_guard' in data:world.__class__=FactorialWorld
    return world


def branch(base,target,event,method):
    if method not in METHODS:raise ValueError('Undeclared method')
    world=fork12(base,target,event,method) if method in ('index_only','labels_guard') else fork(base,target,event,method)
    if fingerprint(world.state)!=fingerprint(base.state):raise ValueError('Actual persistent initial state was not preserved')
    return world


def action_schema(tools):
    branches=[]
    for tool in tools:
        f=tool['function']
        branches.append({'type':'object','properties':{
            'name':{'type':'string','enum':[f['name']]},'arguments':copy.deepcopy(f['parameters'])},
            'required':['name','arguments'],'additionalProperties':False})
    if not branches:raise ValueError('No public tools')
    return {'type':'object','properties':{'action':{'anyOf':branches}},'required':['action'],'additionalProperties':False}


def messages_for(world):
    instruction=POLICY+'''\n本实验所有模型使用同一串行JSON动作接口。每轮只返回一个JSON对象：
{"action":{"name":"公开工具名","arguments":{工具参数}}}。
每轮只做一个工具动作，工具回包在下一条user消息的tool_result字段中。公开工具的含义与参数如下。
不要输出推理过程，不在文字中宣称已提交；必须实际调用submit_answer。工具不会返回标准答案。'''
    return [{'role':'user','content':instruction+'\n'+json.dumps(
        {'public_tools':world.tools(),'task_input':world.initial_input()},ensure_ascii=False)}]


def body(messages,tools,model,tokenizer,settings):
    count=count_chat_tokens(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))
    if count+settings['max_output_tokens']>settings['context_limit']:
        raise ValueError('context_limit_no_silent_truncation')
    return {'model':model,'messages':copy.deepcopy(messages),'stream':False,
        'temperature':settings['temperature'],'seed':settings['seed'],'max_tokens':settings['max_output_tokens'],
        'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema',
            'json_schema':{'name':'serial_public_action','strict':True,'schema':action_schema(tools)}}},count


def post(port,request):
    req=urllib.request.Request('http://127.0.0.1:'+str(port)+'/v1/chat/completions',
        data=json.dumps(request,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(NoRedirect()).open(req,timeout=1800) as response:return json.load(response)


def decode(raw):
    choice=raw['choices'][0]
    if choice['finish_reason']=='length':raise ValueError('output_limit_not_content_error')
    if choice['finish_reason']!='stop':raise ValueError('response_not_completed')
    message=choice['message']
    if message.get('role')!='assistant':raise ValueError('invalid_assistant_role')
    value=json.loads(message['content'])
    if not isinstance(value,dict) or set(value)!={'action'}:raise ValueError('invalid_action_envelope')
    action=value['action']
    if not isinstance(action,dict) or set(action)!={'name','arguments'} or not isinstance(action['name'],str) or not isinstance(action['arguments'],dict):
        raise ValueError('invalid_action_shape')
    return action


def phase(world,folder,port,model,tokenizer,settings):
    started=time.monotonic();start_hash=fingerprint(world.state);public_hash=fingerprint(world.public)
    checkpoint=folder/'checkpoint.private.json';calls=0;elapsed=0;next_action=1;status='action_limit'
    usage={'input_tokens':0,'output_tokens':0};chat=messages_for(world)
    if checkpoint.exists():
        old=read(checkpoint)
        if old['start_hash']!=start_hash or old['public_hash']!=public_hash:raise ValueError('Checkpoint actual start differs')
        world=restore(old['world']);chat=old['messages'];calls=old['requests'];elapsed=old['seconds']
        next_action=old['next_action'];status=old['status'];usage=old['usage']
        if old['terminal']:return world,status,calls,elapsed,usage
    def save(terminal=False):
        dump(checkpoint,{'start_hash':start_hash,'public_hash':public_hash,'world':prior.snapshot(world),
            'messages':chat,'requests':calls,'seconds':elapsed+time.monotonic()-started,'next_action':next_action,
            'status':status,'terminal':terminal,'usage':usage,'harness':'shared_serial_json_action14'})
    save()
    for number in range(next_action,settings['max_actions']+1):
        response=folder/('response_%03d.private.json'%number);pending=folder/('dispatch_%03d.json'%number)
        try:
            if response.exists():raw=read(response);calls=max(calls,number)
            elif pending.exists():
                # Count the dispatched attempt even when the response was lost.
                # Never repeat an ambiguous request to improve coverage.
                calls=max(calls,number);status='indeterminate_inflight_no_resample';break
            else:
                request,ntokens=body(chat,world.tools(),model,tokenizer,settings)
                dump(folder/('request_%03d.private.json'%number),request)
                dump(pending,{'input_tokens':ntokens,'request_sha256':fingerprint(request),'dispatch_time':time.time()})
                calls+=1;raw=post(port,request);dump(response,raw)
            action=decode(raw)
            raw_usage=raw.get('usage') or {}
            usage['input_tokens']+=raw_usage.get('prompt_tokens',0) or 0
            usage['output_tokens']+=raw_usage.get('completion_tokens',0) or 0
            result=world.call(action['name'],action['arguments'])
            chat.append({'role':'assistant','content':json.dumps({'action':action},ensure_ascii=False)})
            chat.append({'role':'user','content':json.dumps({'tool_result':{'name':action['name'],'result':result}},ensure_ascii=False)})
            next_action=number+1
            if world.done:status='submitted';break
            save()
        except urllib.error.HTTPError as error:
            dump(folder/('http_error_%03d.private.json'%number),{'status':error.code,'body':error.read(12000).decode(errors='replace')})
            status='http_'+str(error.code);break
        except Exception as error:
            known=('context_limit_no_silent_truncation','output_limit_not_content_error','response_not_completed',
                'invalid_assistant_role','invalid_action_envelope','invalid_action_shape')
            status=str(error) if isinstance(error,ValueError) and str(error) in known else 'runner_or_transport_'+type(error).__name__
            break
    save(True)
    return world,status,calls,elapsed+time.monotonic()-started,usage


def one(actor,event,method,out,port,model,tokenizer,settings,base=None,world_factory=World):
    sid=actor['source_id'];eid='initial' if event is None else event['event_id']
    rowpath=out/'rows'/sid/(eid+'__'+method+'.json')
    checkpoint_hash=fingerprint(prior.snapshot(base)) if base is not None else None
    if rowpath.exists():
        row=read(rowpath)
        if row.get('initial_checkpoint_hash')!=checkpoint_hash:raise ValueError('Existing row actual base differs')
        if event is None:
            checkpoint=read(out/'phases'/sid/(eid+'__'+method)/'checkpoint.private.json')
            return row,restore(checkpoint['world'])
        return row,None
    row={'source_id':sid,'event_id':eid,'arm':method,'initial_checkpoint_hash':checkpoint_hash,
        'harness':'shared_serial_json_action14'}
    if base is not None and not base.done:
        row.update(submitted=False,status='blocked_actual_initial_not_submitted',requests=0,seconds=0,
            usage={'input_tokens':0,'output_tokens':0})
        dump(rowpath,row);return row,None
    world=world_factory(actor['initial']) if event is None else branch(copy.deepcopy(base),event['controller'],eid,method)
    start_hash=fingerprint(world.state)
    world,status,calls,seconds,usage=phase(world,out/'phases'/sid/(eid+'__'+method),port,model,tokenizer,settings)
    row.update(submitted=world.done,status=status,requests=calls,seconds=seconds,usage=usage,
        pre_state_hash=start_hash,public_hash=fingerprint(world.public),environment=world.public['environment'],
        **export_observations(world))
    dump(rowpath,row);return row,world
