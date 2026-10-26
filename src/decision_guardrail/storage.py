"""Local and S3 staging; credentials use the standard AWS provider chain."""
import hashlib
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
import uuid
from .utils import read_json, write_json


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 object/prefix URI without accepting ambiguous query strings."""
    p = urlparse(uri)
    if p.scheme != 's3' or not p.netloc or p.query or p.fragment or '@' in p.netloc:
        raise ValueError(f'Invalid S3 URI: {uri}')
    return p.netloc, p.path.lstrip('/')


def safe_member(root: Path, name: str) -> Path:
    """Reject absolute paths, traversal, and symlink escapes."""
    if not name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name:
        raise ValueError('Unsafe storage member')
    target = (root/name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Storage member escapes destination')
    return target


def stage_s3(uri: str, cache: Path, client=None, region: str | None = None) -> Path:
    """Stage all objects under a prefix with atomic files and an ETag/size manifest.

    ETags are change indicators, not cryptographic integrity proofs. Existing files
    are additionally checked by SHA-256. Concurrent jobs should use distinct caches.
    """
    if client is None:
        import boto3
        client = boto3.client('s3', region_name=region)
    bucket, key = parse_s3_uri(uri)
    prefix = key.rstrip('/') + '/' if key else ''
    root = cache/hashlib.sha256(uri.encode()).hexdigest()[:20]
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'.s3-manifest.json'
    old = read_json(manifest_path) if manifest_path.exists() else {}
    current = {}
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('/'):
                continue
            relative = obj['Key'][len(prefix):]
            destination = safe_member(root,relative)
            signature = {'etag':obj.get('ETag'), 'size':obj['Size']}
            prior = old.get(relative,{})
            if destination.is_file() and all(prior.get(k)==v for k,v in signature.items()) and destination.stat().st_size==obj['Size'] and prior.get('sha256')==file_hash(destination):
                current[relative] = prior
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.part')
            try:
                client.download_file(bucket,obj['Key'],str(temporary))
                if temporary.stat().st_size != obj['Size']:
                    raise IOError('Incomplete S3 object: '+relative)
                signature['sha256'] = file_hash(temporary)
                os.replace(temporary,destination)
            finally:
                temporary.unlink(missing_ok=True)
            current[relative] = signature
    if not current:
        raise FileNotFoundError('No S3 objects found at '+uri)
    for stale in old.keys()-current.keys():
        safe_member(root,stale).unlink(missing_ok=True)
    write_json(manifest_path,current)
    return root


def file_hash(path: Path) -> str:
    """Stream SHA-256 without loading model shards into memory."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_model_files(root: Path) -> dict:
    """Validate original multimodal Gemma structure and every indexed shard."""
    if not root.is_dir():
        raise FileNotFoundError(f'Model directory missing: {root}; stage the original BF16 model')
    config = read_json(root/'config.json')
    if config.get('model_type') != 'gemma4' or not config.get('vision_config') or not config.get('text_config'):
        raise ValueError('Expected complete multimodal Gemma 4 config (vision_config/text_config)')
    if config['text_config'].get('hidden_size')!=2560 or config['text_config'].get('num_hidden_layers')!=42 or not config.get('audio_config'):
        raise ValueError('Expected original E4B text dimensions and complete audio/vision structure')
    if config.get('quantization_config') or config.get('text_config',{}).get('quantization_config'):
        raise ValueError('Quantised base checkpoint rejected')
    for key in ['dtype','torch_dtype']:
        dtype = config.get(key, config.get('text_config',{}).get(key))
        if dtype and dtype not in {'bfloat16','torch.bfloat16'}:
            raise ValueError('Base checkpoint metadata must specify BF16')
    required = ['tokenizer_config.json']
    for name in required:
        if not (root/name).is_file():
            raise FileNotFoundError(f'Missing model asset: {name}')
    if not (root/'preprocessor_config.json').is_file():
        processor_path = root/'processor_config.json'
        if not processor_path.is_file():
            raise FileNotFoundError('Missing model asset: processor_config.json or preprocessor_config.json')
        processor = read_json(processor_path)
        if processor.get('processor_class') != 'Gemma4Processor' or not processor.get('image_processor') or not processor.get('feature_extractor'):
            raise ValueError('Expected complete Gemma4Processor image and audio settings')
    if not any((root/name).is_file() for name in ['tokenizer.json','tokenizer.model']):
        raise FileNotFoundError('Missing tokenizer.json or tokenizer.model')
    index = root/'model.safetensors.index.json'
    if index.exists():
        mapping = read_json(index).get('weight_map',{})
        if not mapping:
            raise ValueError('Empty model shard index')
        shards = sorted(set(mapping.values()))
    else:
        shards = ['model.safetensors']
    for name in shards:
        path = safe_member(root,name)
        if not path.is_file() or path.stat().st_size==0:
            raise FileNotFoundError(f'Missing or empty shard: {name}')
    return {'config':config, 'shards':shards, 'total_bytes':sum(safe_member(root,n).stat().st_size for n in shards)}


def base_identity(root: Path) -> dict:
    """Identify a base by config and full shard digests (may take minutes)."""
    info = validate_model_files(root)
    return {'config_sha256':file_hash(root/'config.json'), 'shards':{name:file_hash(safe_member(root,name)) for name in info['shards']}}


def resolve_model(config: dict) -> Path:
    """Resolve only approved local/S3 sources and validate before model imports."""
    m = config['model']
    root = Path(m['local_path']) if m['source']=='local' else stage_s3(m['s3_uri'],Path(m['cache_dir']),region=config['storage']['region'])
    validate_model_files(root)
    return root.resolve()


def stage_s3_file(uri: str, cache: Path, client=None, region: str | None = None) -> Path:
    """Atomically stage exactly one dataset object, never its parent prefix."""
    if client is None:
        import boto3
        client=boto3.client('s3',region_name=region)
    bucket,key=parse_s3_uri(uri)
    if not key or key.endswith('/'):
        raise ValueError('Dataset URI must name an object')
    root=cache/hashlib.sha256(uri.encode()).hexdigest()[:20]
    root.mkdir(parents=True,exist_ok=True)
    destination=safe_member(root,PurePosixPath(key).name)
    info=client.head_object(Bucket=bucket,Key=key)
    signature={'etag':info.get('ETag'),'size':info['ContentLength']}
    manifest=root/'.object.json'
    old=read_json(manifest) if manifest.exists() else {}
    if destination.is_file() and all(old.get(k)==v for k,v in signature.items()) and destination.stat().st_size==signature['size'] and old.get('sha256')==file_hash(destination):
        return destination
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.part')
    try:
        client.download_file(bucket,key,str(temporary))
        if temporary.stat().st_size!=signature['size']:
            raise IOError('Incomplete S3 dataset object')
        signature['sha256']=file_hash(temporary)
        os.replace(temporary,destination)
        write_json(manifest,signature)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
