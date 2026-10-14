"""Find which Claude models the Bedrock planner can actually call on this account.

    python deploy/check_bedrock.py                 # region from AWS_REGION (default eu-west-2)
    python deploy/check_bedrock.py --region us-east-1

Lists the Anthropic inference profiles visible in the region, then sends each
a tiny Converse request shaped like the planner's (JSON reply, few tokens).
Pick a working ID and deploy with REVIVE_MODEL_ID=<id>.
"""
import argparse
import json
import os
import time

import boto3

PROMPT = ('Reply with JSON only: {"tool": "done", "reason": "test"}')


def candidates(bedrock):
    ids = []
    try:
        pages = bedrock.get_paginator("list_inference_profiles").paginate(typeEquals="SYSTEM_DEFINED")
        for page in pages:
            ids += [p["inferenceProfileId"] for p in page["inferenceProfileSummaries"]
                    if "anthropic" in p["inferenceProfileId"]]
    except Exception as e:
        print(f"(could not list inference profiles: {type(e).__name__}: {e})")
    try:
        models = bedrock.list_foundation_models(byProvider="Anthropic")["modelSummaries"]
        ids += [m["modelId"] for m in models if "ON_DEMAND" in m.get("inferenceTypesSupported", [])]
    except Exception as e:
        print(f"(could not list foundation models: {type(e).__name__}: {e})")
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-west-2"))
    ap.add_argument("--only", help="substring filter, e.g. haiku")
    args = ap.parse_args()

    bedrock = boto3.client("bedrock", region_name=args.region)
    runtime = boto3.client("bedrock-runtime", region_name=args.region)
    ids = [i for i in candidates(bedrock) if not args.only or args.only in i]
    print(f"region {args.region}: {len(ids)} Anthropic model/profile IDs\n")

    for model_id in ids:
        t0 = time.perf_counter()
        try:
            resp = runtime.converse(modelId=model_id,
                                    messages=[{"role": "user", "content": [{"text": PROMPT}]}],
                                    inferenceConfig={"maxTokens": 300})
            text = next((c["text"] for c in resp["output"]["message"]["content"] if "text" in c), "")
            json.loads(text[text.find("{"): text.rfind("}") + 1])
            print(f"  OK    {model_id}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        except Exception as e:
            print(f"  FAIL  {model_id}  {type(e).__name__}: {str(e)[:110]}")


if __name__ == "__main__":
    main()
