"""Match local Qwen chat-template tool arguments without changing wire semantics.

The HTTP protocol uses JSON strings for function arguments; the checkpoint
tokenizer template requires a mapping. Preflight converts a COPY, never tool data.
"""
import copy
import json
import urllib.request

from token_count import count_chat_tokens


def tokenizer_messages(messages):
    result=copy.deepcopy(messages)
    for message in result:
        if message['role']=='assistant':
            if message.get('content') is None:message['content']=''
            if 'reasoning' in message and 'reasoning_content' not in message:message['reasoning_content']=message.pop('reasoning')
            for call in message.get('tool_calls',[]):
                function=call['function'];arguments=function.get('arguments',{})
                if isinstance(arguments,str):arguments=json.loads(arguments)
                if not isinstance(arguments,dict):raise ValueError('Tool arguments must be a JSON object')
                function['arguments']=arguments
    return result


def make_request(base,model,token,messages,tools,tokenizer):
    template_messages=tokenizer_messages(messages)
    count=count_chat_tokens(tokenizer.apply_chat_template(template_messages,tools=tools,tokenize=True,add_generation_prompt=True))
    if count+8192>65536:raise ValueError('context_limit_no_silent_truncation')
    wire=copy.deepcopy(messages)
    for message in wire:
        if message['role']=='assistant' and message.get('content') is None:message['content']=''
    body={'model':model,'messages':wire,'tools':tools,'tool_choice':'auto','parallel_tool_calls':False,
        'temperature':1,'reasoning_effort':'medium','max_tokens':8192,'stream':False}
    request=urllib.request.Request(base+'/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+token},method='POST')
    return request,body,count
