#!/usr/bin/env bash
# Build the ARM64 image, push it to ECR and deploy/update the ReVive stack.
#
#   ./deploy/deploy.sh                 # first run creates everything (~10 min, mostly CloudFront)
#   ./deploy/deploy.sh                 # later runs push a new image and restart the app (~2 min)
#
# Env: AWS_REGION (default eu-west-2), REVIVE_STACK (default revive),
#      REVIVE_MODEL_ID (Bedrock model / inference profile for the planner),
#      REVIVE_INSTANCE_TYPE (default t4g.medium)
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-eu-west-2}"
STACK="${REVIVE_STACK:-revive}"
REPO="revive"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
TAG="$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo "-dirty-$(date +%s)" || true)"
IMAGE="$REGISTRY/$REPO:$TAG"
echo "==> account $ACCOUNT · region $REGION · image $IMAGE"

# ---- 1. container registry ----
if ! aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1; then
  echo "==> creating ECR repository $REPO"
  aws ecr create-repository --repository-name "$REPO" --image-scanning-configuration scanOnPush=true >/dev/null
  aws ecr put-lifecycle-policy --repository-name "$REPO" --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep last 5","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":5},"action":{"type":"expire"}}]}' >/dev/null
fi

# ---- 2. build + push (ARM64 for Graviton) ----
echo "==> building $IMAGE (linux/arm64)"
docker build --platform linux/arm64 -t "$IMAGE" .
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"
docker push "$IMAGE"

# ---- 3. stack ----
PREFIX_LIST=$(aws ec2 describe-managed-prefix-lists \
  --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
  --query 'PrefixLists[0].PrefixListId' --output text)

PARAMS=("CloudFrontPrefixList=$PREFIX_LIST")
[ -n "${REVIVE_MODEL_ID:-}" ] && PARAMS+=("ModelId=$REVIVE_MODEL_ID")
[ -n "${REVIVE_INSTANCE_TYPE:-}" ] && PARAMS+=("InstanceType=$REVIVE_INSTANCE_TYPE")

if aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1; then
  echo "==> updating stack $STACK (template changes only; the image is rolled out below)"
  aws cloudformation deploy --stack-name "$STACK" --template-file deploy/cloudformation.yaml \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --parameter-overrides "${PARAMS[@]}"

  INSTANCE=$(aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
  echo "==> rolling out $IMAGE on $INSTANCE via SSM"
  CMD=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
    --parameters "commands=[\"/usr/local/bin/revive-update $IMAGE\"]" \
    --query Command.CommandId --output text)
  aws ssm wait command-executed --command-id "$CMD" --instance-id "$INSTANCE" || {
    aws ssm get-command-invocation --command-id "$CMD" --instance-id "$INSTANCE" \
      --query '[Status,StandardErrorContent]' --output text; exit 1; }
else
  echo "==> creating stack $STACK (CloudFront takes ~5-10 minutes)"
  aws cloudformation deploy --stack-name "$STACK" --template-file deploy/cloudformation.yaml \
    --capabilities CAPABILITY_IAM --parameter-overrides "ImageUri=$IMAGE" "${PARAMS[@]}"
fi

URL=$(aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='Url'].OutputValue" --output text)

# ---- 4. wait until the app answers through CloudFront ----
echo "==> waiting for $URL/api/health"
for _ in $(seq 60); do
  if curl -fsS "$URL/api/health" >/dev/null 2>&1; then
    echo "==> ReVive is live: $URL"
    exit 0
  fi
  sleep 10
done
echo "!! $URL did not answer yet. The instance may still be booting; check: aws ssm start-session --target <instance-id>"
exit 1
