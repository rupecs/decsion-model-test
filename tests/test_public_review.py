from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_public_tree import review

def test_sensitive_operational_material_is_flagged_without_printing_values(tmp_path):
 text='AWS_ACCESS_KEY_ID='+('AKIA'+'A'*16)+'\n'+('/'+'Users'+'/someone/private/')+'\n'+'server='+'.'.join(['63','141','33','8'])
 (tmp_path/'bad.txt').write_text(text)
 result=review(tmp_path)
 assert {'aws_access_key','workstation_home','nonloopback_ipv4_review'}<=set(x['rule'] for x in result['findings'])
 assert 'A'*16 not in str(result['findings'])

def test_four_part_package_versions_are_not_treated_as_server_addresses(tmp_path):
 (tmp_path/'tested_gpu_constraints.txt').write_text('nvidia-example-cu12=='+'.'.join(['11','3','3','83'])+'\n')
 assert review(tmp_path)['findings']==[]

def test_environment_files_are_disallowed_even_without_detectable_secret(tmp_path):
 (tmp_path/'.env').write_text('REGION=example\n')
 assert any(x['rule']=='operational_or_model_file' for x in review(tmp_path)['findings'])
