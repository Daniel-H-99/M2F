python transfer_motion.py --data_dir data/tmp/test.mp4 --driving_dir data/tmp/test.mp4
python fillout_mesh.py --data_dir data/tmp/test.mp4 --checkpoint '/home/server25/minyeong_workspace/M2F/log/kkj-256 27_01_22_04.11.58/00000059-checkpoint.pth.tar'
python inference.py --data_dir data/tmp/test.mp4 --checkpoint '/home/server25/minyeong_workspace/M2F/log/kkj-256 27_01_22_04.11.58/00000059-checkpoint.pth.tar'
python data/dataset/paste_patch.py --data_dir data/tmp/test.mp4 --patch_dir demo_img