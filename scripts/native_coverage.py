"""GPU-only official tokenizer coverage before any optimisation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from decision_guardrail.environment import set_offline
set_offline()
from decision_guardrail.config import load_config
from decision_guardrail.training_readiness import assert_training_ready
from decision_guardrail.environment import require_gpu
from decision_guardrail.model_loader import load_model
from decision_guardrail.dataset import load_splits,build_dataset
from decision_guardrail.utils import write_json
compute,report=sys.argv[1],Path(sys.argv[2])
c=load_config('sagemaker',compute,'streaming_context_v14_development',[f'paths.report_dir={report}',f'paths.output_dir={report}/unused-coverage-output'])
assert_training_ready(c)
from decision_guardrail.storage import resolve_model,base_identity
identity=base_identity(resolve_model(c))
if identity!=json.loads((ROOT/'reports/reference_base_identity.json').read_text()):raise RuntimeError('Staged original base differs from the reference model; no GPU fallback is allowed')
write_json(report/'verified_original_base.json',identity)
environment=require_gpu(c);model,processor=load_model(c)
coverage={}
for split,rows in load_splits(c).items():
 item={};encoded=build_dataset(rows,processor,model,item);coverage[split]=item
 assert len(encoded)==c['dataset']['splits'][split] and item['skipped']==item['truncated']==0
 del encoded
assert sum(x['records'] for x in coverage.values())==17467
write_json(report/'native_token_coverage.json',coverage);write_json(report/'gpu_environment.json',environment)
print(json.dumps(coverage,indent=2))
