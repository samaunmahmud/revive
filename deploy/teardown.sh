#!/usr/bin/env bash
# Delete everything deploy.sh created: the stack (EC2, CloudFront, S3, IAM, VPC) and the ECR repo.
#   ./deploy/teardown.sh
set -euo pipefail
REGION="${AWS_REGION:-eu-west-2}"
STACK="${REVIVE_STACK:-revive}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text 2>/dev/null || true)
if [ -n "$BUCKET" ] && [ "$BUCKET" != "None" ]; then
  echo "==> emptying s3://$BUCKET (CloudFormation can't delete a non-empty bucket)"
  aws s3 rm "s3://$BUCKET" --recursive --quiet
fi

echo "==> deleting stack $STACK (CloudFront teardown takes a few minutes)"
aws cloudformation delete-stack --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --stack-name "$STACK"

echo "==> deleting ECR repository revive"
aws ecr delete-repository --repository-name revive --force >/dev/null 2>&1 || true
echo "==> done; nothing from ReVive is left running"
