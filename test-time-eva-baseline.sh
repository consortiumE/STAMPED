#DATASETS=("cifar10c_8_2" "cifar100c_80_20" "imagenet_c_places" "imagenet_c_text" "cifar10_c_noise" "cifar10_c_svhn" "cifar100_c_lsun" "cifar10_c_tiny" "cifar100_c_noise" "cifar100_c_svhn" "cifar100_c_lsun" "cifar100_c_tiny")
DATASETS=("cifar10_c_svhn" "cifar10_c_lsun" "cifar10_c_tiny" "cifar10_c_noise")

#METHODS=("source" "norm_test" "cotta" "eata" "tent" "sar" "rotta" "owttt" "sotta" "stamp")
METHODS=("stamp")

GPU_id=2
test-baseline(){
  DATASET=$1
  METHOD=$2
  GPU_id=$(($3 % 8))
  output_dir="test-time-evaluation-ECCV/${DATASET}/${METHOD}"
  cfg="cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml"
  if [ "$METHOD" == "stamp" ]; then
    if echo "$DATASET" | grep -q "cifar100_c";then
      lr=0.05
      alpha=0.9
    elif echo "$DATASET" | grep -q "cifar10_c";then
      lr=0.1
      alpha=0.25
    elif echo "$DATASET" | grep -q "imagenet";then
      lr=0.01
      alpha=0.8
    elif echo "$DATASET" | grep -q "cifar100c_80_20";then
      lr=0.1
      alpha=0.6
    elif echo "$DATASET" | grep -q "cifar10c_8_2";then
      lr=0.1
      alpha=0.25
    fi
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/cifar10c_8_2/stamp.yaml --output_dir output/STAMP_GMED/cifar10c_8_2 \
      --OPTIM_LR "lr" --STAMP_ALPHA "alpha" &
  elif [ "$METHOD" == "source" ]; then
    CUDA_VISIBLE_DEVICES="$GPU_id" python test-time-evaluation.py --cfg cfgs/Online_TTA_os/${DATASET}/${METHOD}.yaml --output_dir "${output_dir}" &
  fi
}

for DATASET in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    test-baseline $DATASET $METHOD $GPU_id
  GPU_id=$((GPU_id + 1))
  done
done