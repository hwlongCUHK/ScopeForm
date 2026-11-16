"""Matched Qwen evaluation with explicit native decoding and state-based repair."""
import argparse, copy, hashlib, importlib, json, os, shutil, sys, threading, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone
BASE=Path(__file__).resolve().parent
os.environ['CPM_BASELINE_ROOT']=str(BASE/'baseline')
os.environ['CPM_PACKAGE_ROOT']=str(BASE/'package')
sys.path.insert(0,str(BASE))
import run_five_settings as five

def dump(p,v):
 five.dump(p,v)

def reuse_native_prefixes(out, source, initial_run, actors, selected_settings, config, run):
 """Freeze first reading/generation prefixes; repair and Guard continue live."""
 previous=five.read(source/'config.json')
 assert five.read(source/'run_status.json')['status']=='complete', 'Prefix cohort must be complete'
 for key in ('model','actors_sha256','runtime_code_sha256','original_baseline_sha256'):
  assert previous[key]==config[key], 'Prefix source differs: '+key
 assert previous.get('native_tool_choice','auto')==config['native_tool_choice']
 for name in ('run_methods.py','common.py','methods/frame_protocol.py','definitions/frames.json'):
  assert previous['code_sha256'][name]==config['code_sha256'][name], 'Native prompt/loop differs: '+name
 entries=[]
 for actor in actors:
  sid=actor['source_id'];initial=Path('main/shared/phases')/sid/'initial__initial/checkpoint.private.json'
  assert five.read(source/initial)['world']==five.read(initial_run/initial)['world']
  for setting in selected_settings:
   arm=setting+'__formalization'
   for event in actor['events']:
    rel=Path(arm)/'phases'/sid/(event['event_id']+'__'+five.SETTINGS[setting])
    origin=source/'main'/rel;destination=out/'main'/rel
    cp=five.read(origin/'checkpoint.private.json');assert cp['terminal']
    files=sorted(origin.glob('call_*.attempts.json'));through=None
    for f in files:
     n=int(f.name.split('_')[1].split('.')[0]);response=origin/f'call_{n:03d}.response.private.json'
     if response.exists():
      choice=five.read(response)['choices'][0]
      if any(c['function']['name']=='solve_and_submit' for c in choice['message'].get('tool_calls',[])):
       through=n;break
    # Native failures are also replayed, not replaced by a favorable new draw.
    if through is None:through=max(int(f.name.split('_')[1].split('.')[0]) for f in files)
    names=[];charged=0
    for f in files:
     n=int(f.name.split('_')[1].split('.')[0])
     if n>through:continue
     attempts=five.read(f);assert all(v['stage']=='native' for v in attempts)
     charged+=len(attempts)
     for suffix in ('request.private.json','response.private.json','attempts.json'):
      filename=f'call_{n:03d}.{suffix}'
      if (origin/filename).exists():names.append(filename)
    destination.mkdir(parents=True,exist_ok=True)
    for filename in names:
     src=origin/filename;dst=destination/filename
     if not dst.exists():shutil.copy2(src,dst)
     assert run.sha(src)==run.sha(dst), 'Cached prefix differs: '+str(dst)
    entry={'source_id':sid,'event_id':event['event_id'],'arm':arm,'phase_relative_path':str(rel),
           'through_logical_request':through,'charged_reused_calls':charged,
           'files':{name:run.sha(origin/name) for name in names}}
    entries.append(entry);dump(destination/'native_prefix_manifest.json',entry)
 return {'run':str(source),'config_sha256':run.sha(source/'config.json'),'entries':entries,
         'charged_reused_calls':sum(x['charged_reused_calls'] for x in entries),
         'protocol':'All first native prefixes reused, including terminal native failures; post-boundary calls new; cached physical calls remain charged'}

def wire_payload(payload, choice):
 sent=copy.deepcopy(payload);sent.pop('thinking',None)
 sent['chat_template_kwargs']={'enable_thinking':False};sent['seed']=20261022
 if sent.get('tools'):sent['tool_choice']=choice
 return sent

def source_endpoint_map(source_ids, ports):
 """Public source identity fixes the replica for every event and both arms."""
 return {sid:f'http://127.0.0.1:{ports[int(hashlib.sha256(sid.encode()).hexdigest(),16)%len(ports)]}/v1'
         for sid in sorted(source_ids)}

def routed_client_class(baseline, out, choice, endpoints):
 class Client(baseline.Client):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.routing=threading.local()
  def route_source(self,source_id,arm,event_id):
   self.routing.value={'source_id':source_id,'arm':arm,'event_id':event_id,'endpoint':endpoints[source_id]}
  def request(self,path,payload=None):
   req=urllib.request.Request(self.routing.value['endpoint']+path,data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None,headers={'Content-Type':'application/json'})
   try:
    with urllib.request.build_opener(baseline.NoRedirect()).open(req,timeout=1800) as response:return json.load(response)
   except urllib.error.HTTPError as error:
    dump(out/'transport_errors'/f'{time.time_ns()}.json',{'status':error.code,'body':error.read().decode(errors='replace')})
    raise RuntimeError('api_http_'+str(error.code)) from None
  def chat(self,payload):
   sent=wire_payload(payload,choice)
   stamp=str(time.time_ns())+'_'+str(threading.get_ident())
   dump(out/'transport_requests'/f'{stamp}.json',sent)
   dump(out/'transport_routes'/f'{stamp}.json',self.routing.value)
   return super().chat(sent)
 return Client

def wire_signatures(source, fingerprint):
 signatures=set()
 for path in (source/'transport_requests').glob('*.json'):
  try:payload=five.read(path)
  except json.JSONDecodeError:continue  # An unrelated concurrent phase may still be writing.
  signatures.add(fingerprint(payload))
 return signatures

def reuse_baseline(out, source, initial_run, actors, selected_settings, config, run):
 """Reuse terminal controls only when the same native decoding was actually sent."""
 choice=config['native_tool_choice']
 previous=five.read(source/'config.json')
 assert previous['model']==config['model'], 'Baseline model differs'
 assert previous.get('native_tool_choice','auto')==choice, 'Reused baseline native tool choice differs'
 for field in ('actors_sha256','runtime_code_sha256','original_baseline_sha256'):
  assert previous[field]==config[field], 'Baseline input/runtime differs: '+field
 for field in ('run_methods.py','common.py'):
  assert previous['code_sha256'][field]==config['code_sha256'][field], 'Baseline execution differs: '+field
 for field in ('temperature','max_output_tokens','max_actions','max_requests','thinking'):
  assert previous['settings'][field]==config['settings'][field], 'Baseline inference differs: '+field
 actual_wire=wire_signatures(source,run.baseline.fingerprint)
 checked_requests=0
 for actor in actors:
  sid=actor['source_id'];rel=Path('phases')/sid/'initial__initial/checkpoint.private.json'
  old_initial=five.read(source/'main/shared'/rel)
  new_initial=five.read(initial_run/'main/shared'/rel)
  assert run.baseline.fingerprint(old_initial['world'])==run.baseline.fingerprint(new_initial['world']), 'Baseline initial world differs: '+sid
  for setting in selected_settings:
   arm=setting+'__baseline';environment=five.SETTINGS[setting]
   for event in actor['events']:
    folder=source/'main'/arm/'phases'/sid/(event['event_id']+'__'+environment)
    cp=five.read(folder/'checkpoint.private.json')
    assert cp['terminal'], 'Baseline phase not terminal: '+str(folder)
    expected=run.h.branch(run.h.restore(new_initial['world']),event['controller'],event['event_id'],environment)
    assert cp['start_hash']==run.baseline.fingerprint(expected.state), 'Baseline state differs: '+str(folder)
    assert cp['public_hash']==run.baseline.fingerprint(expected.public), 'Baseline public input differs: '+str(folder)
    request=five.read(folder/'call_001.request.private.json')
    assert request['model']==config['model'], 'Baseline request model differs'
    assert request['messages'][0]['content'].endswith(json.dumps(expected.initial_input(),ensure_ascii=False)), 'Baseline task input differs: '+str(folder)
    for request_path in folder.glob('call_*.request.private.json'):
     sent=wire_payload(five.read(request_path),choice)
     assert run.baseline.fingerprint(sent) in actual_wire, 'Baseline actual wire payload differs: '+str(request_path)
     checked_requests+=1
 for setting in selected_settings:
  arm=setting+'__baseline';target=source/'main'/arm;link=out/'main'/arm
  if not link.exists():link.symlink_to(target,target_is_directory=True)
  assert link.resolve()==target.resolve(), 'Reused baseline directory differs'
 return {'run':str(source),'config_sha256':run.sha(source/'config.json'),
         'arms':[s+'__baseline' for s in selected_settings],
         'phases':sum(len(a['events']) for a in actors)*len(selected_settings),
         'native_tool_choice':choice,'actual_wire_logical_requests_verified':checked_requests,
         'same_model_initial_state_and_public_input_checked':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--model',default='Qwen3.8-27B');p.add_argument('--port',type=int,default=18227);p.add_argument('--replica-ports',nargs='+',type=int);p.add_argument('--workers',type=int,default=4);p.add_argument('--scope',choices=['pilot','full'],default='pilot');p.add_argument('--settings',nargs='+',default=['guard']);p.add_argument('--source-ids',nargs='+');p.add_argument('--reuse-initials',type=Path,required=True);p.add_argument('--method',choices=['repair','formalization'],default='repair');p.add_argument('--native-tool-choice',choices=['auto','required'],default='required');p.add_argument('--reuse-baseline',type=Path);p.add_argument('--reuse-native-prefixes',type=Path);a=p.parse_args()
 out=a.run_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 old=five.read(BASE/'source_config.json');run=five.load_frozen(BASE)
 for name,h in old['code_sha256'].items(): assert run.sha(BASE/'frozen_code'/name)==h,name
 for name,h in old['runtime_code_sha256'].items(): assert run.sha(run.baseline.RUNTIME/name)==h,name
 for name,h in old['original_baseline_sha256'].items(): assert run.sha(BASE/'baseline'/name)==h,name
 actors_path=run.baseline.RUNTIME/'confirmation15_inputs_private/actors.private.jsonl'
 assert run.sha(actors_path)==old['actors_sha256']
 actors=[json.loads(x) for x in actors_path.read_text().splitlines() if x]
 if a.source_ids:
  assert a.scope=='pilot', 'Explicit subset must be labeled pilot'
  actors=[x for x in actors if x['source_id'] in a.source_ids]
  assert {x['source_id'] for x in actors}==set(a.source_ids)
 elif a.scope=='pilot':
  actors=[x for modality,count in [('dialogue',3),('drug',2)] for x in sorted([v for v in actors if v['initial']['modality']==modality],key=lambda v:run.baseline.fingerprint(v['source_id']))[:count]]
 actors.sort(key=lambda x:x['source_id'])
 five.ARMS=tuple(s+'__'+v for s in a.settings for v in ['baseline','formalization'])
 settings={**old['settings'],'workers_per_model':a.workers}
 config={**old,'package_root':str(BASE/'package'),'baseline_root':str(BASE/'baseline'),'model':a.model,'endpoint':f'http://127.0.0.1:{a.port}/v1','settings':settings,'source_ids':[x['source_id'] for x in actors],'layout_source_ids':[],'experimental_arms':list(five.ARMS),'suite':'core','scope':a.scope,'formalization_version':'repair','frozen_inference_source':str(BASE/'frozen_code'),'kind':'qwen_matched_formalization','notes_enabled':False,'shared_initials':True,'reused':'No completions reused across models; fresh Qwen initial worlds shared within model','budget':'All physical native and method requests <=40; all actions <=40; failures stay in denominators','integration':'Frozen complete semantic method; original paper setting and Guard','method_temperatures':'temperature=0; enable_thinking=false; output 8192; seed=20261022','transport_timeout_seconds':1800,'transport_adaptation':'DeepSeek thinking field mapped to vLLM chat_template_kwargs.enable_thinking=false; seed 20261022; other inference payload fields unchanged','planned_updates_per_arm':len(actors)*4,'planned_shared_initials':len(actors),'planned_phases':len(actors)*(1+4*len(five.ARMS)),'harness_sha256':run.sha(__file__)}
 initial_run=a.reuse_initials.resolve()
 assert five.read(initial_run/'config.json')['model']==a.model, 'Initial worlds must use the same model'
 config.update(kind='qwen_state_repair', formalization_version=a.method,
   reused='Read-only actual same-model initial worlds; all paired updates are new requests',
   reused_initial_run=str(initial_run), native_tool_choice=a.native_tool_choice,
   integration='Same original paper setting and Guard; '+a.native_tool_choice+' native tool choice for BOTH arms; current-state semantic repair',
   transport_adaptation='enable_thinking=false, seed 20261022; tool_choice='+a.native_tool_choice+' for every native request in BOTH arms',
   source_selection='Three source-format failures (S000130,S000384,S000813) plus lowest-public-fingerprint two dialogue and one drug source; all four events, all failures retained' if a.source_ids else ('Full protocol cohort' if a.scope=='full' else 'Lowest public source fingerprint: three dialogue, two drug'),
   planned_new_phases=len(actors)*4*len(five.ARMS), fresh_model_specific_initials=False)
 ports=a.replica_ports or [a.port]
 config.update(replica_ports=ports,source_endpoint_map=source_endpoint_map(config['source_ids'],ports),
   replica_routing='sha256(UTF-8 public source_id) as integer modulo the ordered weighted port list; both arms and all events share the source replica')
 shared=out/'main/shared';shared.parent.mkdir(parents=True,exist_ok=True)
 if not shared.exists():shared.symlink_to(initial_run/'main/shared',target_is_directory=True)
 assert shared.resolve()==(initial_run/'main/shared').resolve()
 for actor in actors:
  cp=shared/'phases'/actor['source_id']/'initial__initial/checkpoint.private.json'
  assert cp.exists() and five.read(cp)['terminal'], 'Initial phase not yet terminal: '+actor['source_id']
 if a.reuse_baseline:
  reused=reuse_baseline(out,a.reuse_baseline.resolve(),initial_run,actors,a.settings,config,run)
  config.update(reused_baseline=reused,reused_baseline_run=reused['run'],
    planned_new_phases=config['planned_new_phases']-reused['phases'],
    reused='Read-only actual same-model initial worlds and matched '+a.native_tool_choice+' baseline trajectories; formalization updates are new requests')
 if a.reuse_native_prefixes:
  prefixes=reuse_native_prefixes(out,a.reuse_native_prefixes.resolve(),initial_run,actors,a.settings,config,run)
  config.update(reused_native_prefixes=prefixes,
    reused='Same-model initial worlds and baseline; all first native read/generation prefixes copied and charged. Only post-boundary reasoning and Guard continuation are newly generated.',
    latency_note='Native-prefix time is cached; phase wall-time is not an end-to-end latency comparison')
 if (out/'config.json').exists(): assert five.read(out/'config.json')==config
 dump(out/'config.json',config)
 for name in ['run_qwen.py','run_five_settings.py']:shutil.copy2(BASE/name,out/name)
 if not (out/'frozen_code').exists():(out/'frozen_code').symlink_to(BASE/'frozen_code',target_is_directory=True)
 Client=routed_client_class(run.baseline,out,a.native_tool_choice,config['source_endpoint_map'])
 client=Client('local-no-key',config['endpoint'])
 class RoutedRunner(five.runner_class(run)):
  def one(self,actor,event,arm,out,*args,**kwargs):
   self.client.route_source(actor['source_id'],arm if event else 'initial',event['event_id'] if event else 'initial')
   return super().one(actor,event,arm,out,*args,**kwargs)
 runner=RoutedRunner(client,a.model,settings,a.method);run.h.phase=runner.phase
 start=time.monotonic();dump(out/'run_status.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat()})
 runner.workflow(actors,out/'main')
 dump(out/'provider_metadata.json',{'requested_model':a.model,'returned_models':sorted(client.models),'system_fingerprints':sorted(client.fingerprints),'fresh_model_specific_initials':False,'reused_initial_run':str(initial_run),'reused_baseline_run':config.get('reused_baseline_run'),'native_tool_choice':a.native_tool_choice,'replica_ports':ports,'source_endpoint_map':config['source_endpoint_map']})
 dump(out/'run_status.json',{'status':'complete','wall_seconds':time.monotonic()-start,'finished_utc':datetime.now(timezone.utc).isoformat()})
if __name__=='__main__':main()
