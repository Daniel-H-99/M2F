python transfer_motion.py --data_dir data/tmp/test.mp4 --driving_dir data/tmp/test.mp4
python fillout_mesh.py --data_dir data/tmp/test.mp4 --checkpoint 'log/kkj-256 25_01_22_05.54.42/00000079-checkpoint.pth.tar'
python inference.py --data_dir data/tmp/test.mp4 --checkpoint 'log/kkj-256 25_01_22_05.54.42/00000079-checkpoint.pth.tar'
python data/dataset/paste_patch.py --data_dir data/tmp/test.mp4 --patch_dir demo_img