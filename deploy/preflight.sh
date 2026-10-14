#!/usr/bin/env bash
# Check everything deploy.sh and cool_bench.sh need, before spending time on a deploy.
#   ./deploy/preflight.sh
# Prints OK / MISSING for each item and what to do about it. Changes nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
REGION="${AWS_REGION:-eu-west-2}"; export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
ok() { printf '  \033[32mOK\033[0m       %s\n' "$1"; }
bad() { printf '  \033[31mMISSING\033[0m  %s\n           -> %s\n' "$1" "$2"; FAIL=1; }
FAIL=0
echo "ReVive preflight · region $REGION"

command -v aws >/dev/null && ok "AWS CLI $(aws --version 2>&1 | cut -d' ' -f1)" || bad "AWS CLI" "brew install awscli"
if ID=$(aws sts get-caller-identity --query Arn --output text 2>/dev/null); then ok "credentials: $ID"
else bad "AWS credentials" "run: aws configure   (access key, region $REGION, output json)"; fi
docker info >/dev/null 2>&1 && ok "Docker running ($(docker info --format '{{.Architecture}}'))" || bad "Docker" "start Docker Desktop"
command -v gh >/dev/null && ok "GitHub CLI" || bad "GitHub CLI" "brew install gh (only needed to rebuild the report PDF)"

if [ -n "${ID:-}" ]; then
  for t in t4g.medium c8g.large c7i.large; do
    if aws ec2 describe-instance-type-offerings --filters "Name=instance-type,Values=$t" \
         --query 'InstanceTypeOfferings[0].InstanceType' --output text 2>/dev/null | grep -q "$t"; then ok "$t offered in $REGION"
    else bad "$t in $REGION" "pick another region with AWS_REGION=..."; fi
  done
  echo "  ...checking which Claude models Bedrock will run for this account (a few seconds)"
  if .venv/bin/python deploy/check_bedrock.py --region "$REGION" --only haiku 2>/dev/null | tee /tmp/revive-bedrock.txt | grep -q "  OK"; then
    ok "Bedrock: $(grep '  OK' /tmp/revive-bedrock.txt | head -1 | awk '{print $2}')  (use as REVIVE_MODEL_ID)"
  else bad "Bedrock Claude access" "Bedrock console ($REGION) > Model access > enable Anthropic models; the app still works without it (heuristic planner)"; fi
  COOL=$(aws ec2 describe-images --owners aws-marketplace --filters "Name=name,Values=*Optimized*OpenCV*" "Name=architecture,Values=arm64" \
           --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]' --output text 2>/dev/null)
  if [ -n "$COOL" ] && [ "$COOL" != "None" ]; then ok "COOL AMI candidate: $COOL (confirm on the Marketplace launch page)"
  else bad "COOL AMI" "subscribe to 'Cloud Optimized OpenCV For AWS Graviton4' on AWS Marketplace, then pass COOL_AMI=ami-..."; fi
  if aws cloudformation describe-stacks --stack-name "${REVIVE_STACK:-revive}" >/dev/null 2>&1; then ok "stack '${REVIVE_STACK:-revive}' exists"
  else echo "  --       stack not deployed yet (./deploy/deploy.sh creates it)"; fi
fi
[ "$FAIL" = 0 ] && echo "All set: ./deploy/deploy.sh" || echo "Fix the MISSING items above, then re-run."
