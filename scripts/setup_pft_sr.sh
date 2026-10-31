#!/usr/bin/env bash
# Clones PFT-SR (Long et al., CVPR 2025) into third_party/PFT-SR at the commit used
# for the paper and builds its sparse matrix multiplication CUDA op (smm_cuda).
set -e
cd "$(dirname "$0")/.."
PFT_COMMIT=b1dc104bf8a92aa7aae0a89fa983e1f66ad45dda

mkdir -p third_party
if [ ! -d third_party/PFT-SR ]; then
    git clone https://github.com/LabShuHangGU/PFT-SR.git third_party/PFT-SR
fi
cd third_party/PFT-SR
git checkout $PFT_COMMIT
pip install -r requirements.txt
python setup.py develop
cd ops_smm && python setup.py install && cd ..

echo ""
echo "Now download the official PFT weights from"
echo "  https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg"
echo "into third_party/PFT-SR/experiments/pretrained_models/, e.g."
echo "  001_PFT_SRx2_scratch.pth  002_PFT_SRx3_finetune.pth  003_PFT_SRx4_finetune.pth"
