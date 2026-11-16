"""Accept actual tokenizer return shapes; never use len(BatchEncoding) as tokens."""
from collections.abc import Mapping


def count_chat_tokens(encoded):
    if isinstance(encoded,Mapping):
        if 'input_ids' not in encoded:raise ValueError('Tokenizer has no input_ids')
        encoded=encoded['input_ids']
    if hasattr(encoded,'tolist'):encoded=encoded.tolist()
    if not isinstance(encoded,(list,tuple)):raise ValueError('Unsupported tokenizer return type')
    if encoded and isinstance(encoded[0],(list,tuple)):
        if len(encoded)!=1:raise ValueError('Expected one conversation, not an uncounted batch')
        encoded=encoded[0]
    if not encoded or any(type(x) is not int for x in encoded):raise ValueError('Tokenizer did not return integer token IDs')
    return len(encoded)
