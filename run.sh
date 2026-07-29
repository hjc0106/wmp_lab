# cd /home/hongjiacheng/codes/robot_lab/wmp_lab
# unset CUDA_VISIBLE_DEVICES

# python scripts/play_wmp.py \
#   --device cuda:0 \
#   --num_envs 10 \
#   --checkpoint logs/go2_amp_lab_ddp/model_20000.pt \
#   --terrain climb \
#   --num_episodes 1
cd /home/hongjiacheng/codes/robot_lab/wmp_lab
conda run --no-capture-output -n wmp_lab \
    python scripts/visualize_terrains.py --terrain all
# Remote server without display: add --headless --steps 300