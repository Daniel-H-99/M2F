src_dir=data/data_raw
tgt_dir=data/data_preprocessed
home=~/minyeong_workspace/M2F

# start_time="00:00:11"
# end_time="00:10:00"

# echo 'image cropping...'

# cd $home/$src_dir

# for file in *.mp4
# do
#     mkdir -p $home/$tgt_dir/$file
#     mkdir -p $home/$tgt_dir/$file/full
#     mkdir -p $home/$tgt_dir/$file/crop
#     rm -rf $home/$tgt_dir/$file/img
#     ffmpeg -hide_banner -y -i $file -filter:v "crop=540:540:360:0,scale=520x520" -r 25 $home/$tgt_dir/$file/full/%05d.png
#     # ffmpeg -hide_banner -y -ss $start_time -t $end_time -i $file -filter:v "crop=540:540:360:0,scale=520x520" -r 25 $home/$tgt_dir/$file/full/%05d.png
#     CUDA_VISIBLE_DEVICES=1 python $home/data/dataset/crop_portrait.py --data_dir $home/$tgt_dir/$file --crop_level 1.5 --vertical_adjust -0.2 --dest_size 256
#     ffmpeg -loglevel panic -y  -i $file  -strict -2 $home/$tgt_dir/$file/audio.wav
#     ffmpeg -loglevel panic -y  -ss $start_time -i $file -t $end_time -strict -2 $home/$tgt_dir/$file/audio.wav
#     cp $home/$tgt_dir/$file/audio.wav $home/$tgt_dir/$file/_audio.wav
#     # rm -rf $home/$tgt_dir/$file/full
#     mv $home/$tgt_dir/$file/crop $home/$tgt_dir/$file/img
# done

# cd $home/data/dataset

# echo 'pose normalizing...'
# for file in $home/$tgt_dir/*.mp4
# do
#     python pose_normalization.py --data_dir $file --draw_mesh --interpolate_z --ref_path $file/frame_reference.png
# done

