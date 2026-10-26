"""Verify the complete public handover inventory and bytes, without GPU imports."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from review_public_tree import files

def verify():
 manifest=json.loads((ROOT/'package_manifest.json').read_text())
 actual={str(p.relative_to(ROOT)) for p in files() if str(p.relative_to(ROOT))!='package_manifest.json'}
 if actual!=set(manifest['files']):raise ValueError('Public package file inventory changed')
 for name,expected in manifest['files'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT) or (ROOT/name).is_symlink():raise ValueError('Unsafe package path')
  with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  if sha!=expected:raise ValueError('Public package checksum mismatch: '+name)
 return {'public_package_integrity':'passed','files':len(actual),'dataset':'v14','GPU_loaded':False}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
