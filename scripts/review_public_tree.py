"""Local pre-publication screening, including compressed retained/source inputs.

Pattern/file-size screening supplements manual review; it is not a guarantee that
arbitrary future files contain no confidential data. Nothing is uploaded.
"""
import argparse,gzip,hashlib,io,ipaddress,json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','__pycache__','.pytest_cache','.replays','.local','outputs','build','dist'}
RULES={
 'aws_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'private_key':re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
 'github_token':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
 'huggingface_token':re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'),
 'runpod_token':re.compile(r'\brpa_[A-Za-z0-9_-]{25,}\b'),
 'assigned_secret':re.compile(r'''(?i)(?:aws_secret_access_key|api[_-]?key|access[_-]?token|password)\s*[:=]\s*["']([A-Za-z0-9+/=_-]{24,})["']'''),
 'workstation_home':re.compile('/'+'Users'+r'/[^/\s"<>]+/|/'+'home'+r'/(?!user(?:/|\b)|example(?:/|\b))[^/\s"<>]+/'),
 'runpod_ssh_endpoint':re.compile(r'ssh\.runpod\.io'),
}

def files(root=ROOT):
 return sorted(p for p in root.rglob('*') if p.is_file() and not any(x in SKIP or x.endswith('.egg-info') for x in p.relative_to(root).parts))

def review(root=ROOT):
 findings=[];checked=0;expanded=0;total=0;largest=('',0)
 def scan(name,raw):
  nonlocal checked,expanded
  checked+=1
  if raw.startswith(b'PK\x03\x04'):
   with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for item in z.infolist():
     if item.is_dir():continue
     if item.filename.startswith('/') or '..' in Path(item.filename).parts:findings.append({'path':name+'!'+item.filename,'rule':'unsafe_archive_member'})
     if item.file_size>100*1024**2:raise ValueError('Oversized compressed member')
     expanded+=1;scan(name+'!'+item.filename,z.read(item))
   return
  if raw.startswith(b'\x1f\x8b'):
   expanded+=1;data=gzip.decompress(raw)
   if len(data)>100*1024**2:raise ValueError('Oversized compressed member')
   scan(name+'!decompressed',data);return
  try:text=raw.decode('utf-8')
  except UnicodeDecodeError:return
  for rule,pattern in RULES.items():
   for match in pattern.finditer(text):
    # Report location/type only; never print a possible secret value.
    findings.append({'path':name,'rule':rule,'line':text.count('\n',0,match.start())+1})
  for match in re.finditer(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',text):
   try:address=ipaddress.ip_address(match.group())
   except ValueError:continue
   line=text.splitlines()[text.count('\n',0,match.start())]
   if name=='tested_gpu_constraints.txt' and re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]*==[A-Za-z0-9+._-]+',line):continue
   if not address.is_loopback and not address.is_unspecified:
    findings.append({'path':name,'rule':'nonloopback_ipv4_review','line':text.count('\n',0,match.start())+1})
 for path in files(root):
  name=str(path.relative_to(root));size=path.stat().st_size;total+=size
  if size>largest[1]:largest=(name,size)
  if path.is_symlink():findings.append({'path':name,'rule':'symlink_not_allowed'})
  if size>=100*1024**2:findings.append({'path':name,'rule':'github_100MiB_limit'})
  if path.name=='.env' or path.name.startswith('.env.') or any(x in {'.aws','.ssh'} for x in path.relative_to(root).parts) or path.suffix.lower() in {'.pem','.key','.safetensors','.bin'} or path.name in {'session.json','known_hosts','credentials','model.tar.gz','output.tar.gz'}:findings.append({'path':name,'rule':'operational_or_model_file'})
  if re.search(r'data/(?:generated|samples)/(?:streaming_context_v(?:[0-9]|1[0-3])(?:_|/)|streaming_15000|d1_d6)',name):findings.append({'path':name,'rule':'older_dataset'})
  scan(name,path.read_bytes())
 return {'files':len(files(root)),'bytes':total,'largest_file':{'path':largest[0],'bytes':largest[1]},'compressed_members_checked':expanded,'objects_scanned':checked,'findings':findings,'scope':'Local pattern/operational-file screening plus separately documented manual review; not a universal security guarantee','uploads_performed':False}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();result=review()
 if a.output:
  a.output.parent.mkdir(parents=True,exist_ok=True)
  with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));raise SystemExit(bool(result['findings']))
