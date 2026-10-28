# HEART general molecular pretraining -- Uni-Core user module.
#
# Loaded by `unicore-train ... --user-dir ./unimol` from the model root
# (/home/ypan/heart_liver/model_train/heart-molecular-pretraining/model on
# CMRI HPC). Importing the sub-packages
# registers the task/model/architecture/loss used by HEART:
#   task  unimol_contrastive          (tasks/unimol_contrastive.py)
#   model unimol_contrastive, arch unimol_contrastive_base
#   loss  unimol_contrastive
#
# Layout follows LUMI-lab model/unimol/ (GPL-3.0); data/ is taken from
# Uni-Mol unimol/data/ (MIT) because LUMI-lab does not publish its data/
# directory (`data/` is listed in LUMI-lab model/.gitignore). LUMI's
# `import unimol.utils` is dropped: those utilities (docking, conformer
# generation) are not used by pretraining. See THIRD_PARTY_NOTICES.md.
import unimol.tasks
import unimol.data
import unimol.models
import unimol.losses
