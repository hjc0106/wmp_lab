cd /home/hongjiacheng/codes/robot_lab/wmp_lab
unset CUDA_VISIBLE_DEVICES

python scripts/play_wmp.py \
  --device cuda:0 \
  --num_envs 10 \
  --checkpoint logs/go2_amp_lab_ddp/model_20000.pt \
  --terrain climb \
  --num_episodes 1
