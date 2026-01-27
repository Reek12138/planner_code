fuyao deploy \
  --gpus-per-node=1 \
  --nodes=1 \
  --label kespeech_test_vggt\
  --site=fuyao_ppu_c3 \
  --docker-image=infra-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/data-infra/fuyao:chendy3_stepaudio-v4 \
  --project=adc-fm-training \
  --queue=adc-perception-foundation-e2e \
  --experiment=fm_experiment \
  python vggt_encoder.py