import argparse,json
from pathlib import Path
from decision_guardrail.data_generation.context_v14_authoring import write_pack
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(write_pack(Path(__file__).resolve().parents[1],args.output),indent=2))
