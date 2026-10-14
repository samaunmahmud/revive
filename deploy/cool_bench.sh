#!/usr/bin/env bash
# Reproducible COOL benchmark on AWS: stock OpenCV 5 vs COOL on the same Graviton4
# instance, plus stock OpenCV 5 on x86 for reference. One command, self-cleaning.
#
#   COOL_AMI=ami-xxxxxxxx ./deploy/cool_bench.sh
#
# Prerequisites: the ReVive stack is deployed (./deploy/deploy.sh): its S3 bucket,
# instance profile and subnet are reused. You are subscribed to "Cloud Optimized
# OpenCV For AWS Graviton4" on AWS Marketplace; COOL_AMI is that AMI's ID in this
# region (shown on the Marketplace "Launch" page). Each benchmark instance runs
# bench/benchmark.py, uploads JSON to S3 and terminates itself (~15-25 min).
#
# Env: AWS_REGION (default eu-west-2), REVIVE_STACK (default revive),
#      ARM_TYPE (default c8g.large), X86_TYPE (default c7i.large),
#      COOL_SW_PRICE (Marketplace software $/h for ARM_TYPE, default 0.01 for c8g.large)
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-eu-west-2}"; STACK="${REVIVE_STACK:-revive}"
ARM_TYPE="${ARM_TYPE:-c8g.large}"; X86_TYPE="${X86_TYPE:-c7i.large}"; COOL_SW_PRICE="${COOL_SW_PRICE:-0.01}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
: "${COOL_AMI:?set COOL_AMI to the COOL Graviton4 AMI ID for $REGION (Marketplace > your subscription > Launch)}"

out() { aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
res() { aws cloudformation describe-stack-resource --stack-name "$STACK" --logical-resource-id "$1" --query StackResourceDetail.PhysicalResourceId --output text; }
BUCKET=$(out BucketName); PROFILE=$(res InstanceProfile); SUBNET=$(res Subnet)
RUN="bench/$(date +%Y%m%d-%H%M%S)"

for t in "$ARM_TYPE" "$X86_TYPE"; do
  aws ec2 describe-instance-type-offerings --filters "Name=instance-type,Values=$t" --query 'InstanceTypeOfferings[0].InstanceType' --output text | grep -q "$t" \
    || { echo "!! $t is not offered in $REGION"; exit 1; }
done

price() {  # on-demand Linux $/h from the AWS Pricing API (served from us-east-1)
  python3 - "$1" "$REGION" <<'PY'
import json, sys, boto3
itype, region = sys.argv[1], sys.argv[2]
loc = boto3.client("ssm", region_name=region).get_parameter(
    Name=f"/aws/service/global-infrastructure/regions/{region}/longName")["Parameter"]["Value"]
p = boto3.client("pricing", region_name="us-east-1").get_products(ServiceCode="AmazonEC2", Filters=[
    {"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in {
        "instanceType": itype, "location": loc, "operatingSystem": "Linux", "tenancy": "Shared",
        "preInstalledSw": "NA", "capacitystatus": "Used"}.items()])["PriceList"]
terms = json.loads(p[0])["terms"]["OnDemand"]
print(next(iter(next(iter(terms.values()))["priceDimensions"].values()))["pricePerUnit"]["USD"])
PY
}
ARM_PRICE=$(price "$ARM_TYPE"); X86_PRICE=$(price "$X86_TYPE")
COOL_PRICE=$(python3 -c "print(round($ARM_PRICE + $COOL_SW_PRICE, 4))")
echo "==> $ARM_TYPE \$$ARM_PRICE/h (+ COOL \$$COOL_SW_PRICE/h) · $X86_TYPE \$$X86_PRICE/h · results -> s3://$BUCKET/$RUN/"

echo "==> packaging code"
git archive --format=tar.gz -o /tmp/revive-src.tgz HEAD
aws s3 cp /tmp/revive-src.tgz "s3://$BUCKET/$RUN/src.tgz" --quiet

# Shared start of both user-data scripts. \$1 = platform setup that must run first
# (it installs the AWS CLI where missing), then: fetch code, log to S3, always power off.
common() { cat <<EOF
#!/bin/bash
exec > /var/log/revive-bench.log 2>&1
set -x
$1
trap 'aws s3 cp /var/log/revive-bench.log s3://$BUCKET/$RUN/log-\$(hostname).txt; shutdown -h now' EXIT
mkdir -p /opt/revive && cd /opt/revive
aws s3 cp s3://$BUCKET/$RUN/src.tgz . && tar xzf src.tgz
sed '/# web app/,\$d' requirements.txt | grep -v '^boto3' > req-cv.txt     # pinned opencv + numpy only
EOF
}

ARM_UD=$(common "apt-get update -y; apt-get install -y python3.12-venv; command -v aws || snap install aws-cli --classic"; cat <<EOF
# stock OpenCV 5 (pip) in its own venv
python3.12 -m venv /opt/stock && /opt/stock/bin/pip install -q -r req-cv.txt
/opt/stock/bin/python -m bench.benchmark run --label stock-graviton4 --price $ARM_PRICE --out results
# COOL: use COOL's own venv; never install pip OpenCV/numpy into it
COOLPY=/opt/cool/venvs/python_3.12/bin/python
# only benchmark as "cool" if cv2 really is COOL's build (never mislabel stock numbers)
\$COOLPY -c "import cv2; assert '/opt/cool' in cv2.__file__, cv2.__file__; print('COOL', cv2.__version__, cv2.__file__)" \\
  && \$COOLPY -m bench.benchmark run --label cool-graviton4 --price $COOL_PRICE --out results
aws s3 cp results/ s3://$BUCKET/$RUN/results/ --recursive
EOF
)
X86_UD=$(common "dnf install -y python3.12"; cat <<EOF
python3.12 -m venv /opt/stock && /opt/stock/bin/pip install -q -r req-cv.txt
/opt/stock/bin/python -m bench.benchmark run --label stock-x86 --price $X86_PRICE --out results
aws s3 cp results/ s3://$BUCKET/$RUN/results/ --recursive
EOF
)
X86_AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text)

launch() {  # type ami userdata name
  aws ec2 run-instances --instance-type "$1" --image-id "$2" --subnet-id "$SUBNET" --associate-public-ip-address \
    --iam-instance-profile "Name=$PROFILE" --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required --user-data "$3" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$4}]" --query 'Instances[0].InstanceId' --output text
}
ARM_ID=$(launch "$ARM_TYPE" "$COOL_AMI" "$ARM_UD" revive-bench-graviton4)
X86_ID=$(launch "$X86_TYPE" "$X86_AMI" "$X86_UD" revive-bench-x86)
echo "==> launched $ARM_ID ($ARM_TYPE, COOL AMI) and $X86_ID ($X86_TYPE); waiting for results (both self-terminate)"

for _ in $(seq 180); do   # up to 60 min: the Graviton box runs stock and COOL back to back
  n=$(aws s3 ls "s3://$BUCKET/$RUN/results/" 2>/dev/null | grep -c '\.json$' || true)
  [ "$n" -ge 3 ] && break
  sleep 20
done
mkdir -p bench/results
aws s3 cp "s3://$BUCKET/$RUN/results/" bench/results/ --recursive --quiet
aws s3 cp "s3://$BUCKET/$RUN/" bench/results/logs/ --recursive --exclude "*" --include "log-*" --quiet || true
python3 -m bench.benchmark compare bench/results/*.json --baseline stock-graviton4 | tee bench/results/RESULTS.md
aws ec2 terminate-instances --instance-ids "$ARM_ID" "$X86_ID" >/dev/null 2>&1 || true   # in case either hung
echo "==> done: bench/results/RESULTS.md"
