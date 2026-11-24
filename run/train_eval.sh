llamafactory-cli train \
 MT-CSD_Biden_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> 0413_Biden-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Biden_noInstance_Multi-layers-topKL_prompt.py \
 >> 0413_Biden-bs1.log 2>&1 &
wait
llamafactory-cli train \
 MT-CSD_Bitcoin_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> 0413_Bitcoin-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Bitcoin_noInstance_Multi-layers-topKL_prompt.py \
 >> 0413_Bitcoin-bs1.log 2>&1 &
wait
llamafactory-cli train \
 MT-CSD_SpaceX_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> 0413_SpaceX-bs1x.log 2>&1 &
wait
python3 run_multi_predict_SpaceX_noInstance_Multi-layers-topKL_prompt.py \
 >> 0413_SpaceX-bs1.log 2>&1 &
wait
llamafactory-cli train \
 MT-CSD_Tesla_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> 0413_Tesla-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Tesla_noInstance_Multi-layers-topKL_prompt.py \
 >> 0413_Tesla-bs1.log 2>&1 &
wait
llamafactory-cli train \
 MT-CSD_Trump_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> 0413_Trump-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Trump_noInstance_Multi-layers-topKL_prompt.py \
 >> 0413_Trump-bs1.log 2>&1 &
wait



llamafactory-cli train \
 C-MTCSD_Non-marriage_Doctrine_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> Non-marriage_Doctrine-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Non-marriage_Doctrine_noInstance_Multi-layers-topKL_prompt.py \
 >> Non-marriage_Doctrine-bs1.log 2>&1 &
wait

llamafactory-cli train \
 C-MTCSD_Apollo_Go_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> Apollo_Go-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Apollo_Go_noInstance_Multi-layers-topKL_prompt.py \
 >> Apollo_Go-bs1.log 2>&1 &
wait

llamafactory-cli train \
 C-MTCSD_Pre-made_Meals_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> Pre-made_Meals-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Pre-made_Meals_noInstance_Multi-layers-topKL_prompt.py \
 >> Pre-made_Meals-bs1.log 2>&1 &
wait
llamafactory-cli train \
 C-MTCSD_Naked_Resignation_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> Naked_Resignation-bs1x.log 2>&1 &
wait
python3 run_multi_predict_Naked_Resignation_noInstance_Multi-layers-topKL_prompt.py \
 >> Naked_Resignation-bs1.log 2>&1 &
wait

llamafactory-cli train \
 C-MTCSD_iPhone_15_sft_noInstance_custom-loss-multi-layer-topKL_prompt.yaml \
 >> iphone_15-bs1x.log 2>&1 &
wait
python3 run_multi_predict_iPhone_15_noInstance_Multi-layers-topKL_prompt.py \
 >> iphone_15-bs1.log 2>&1 &
wait