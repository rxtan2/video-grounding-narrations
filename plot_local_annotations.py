import os
import argparse
import pickle
import cv2
import json
import sys
import ffmpeg
import numpy as np

parser = argparse.ArgumentParser(description='YouCook2-Interactions visualizations')
parser.add_argument(
        '--video_dir',
        type=str,
        default='/scratch2/youcook2_code/raw_videos/validation_videos/',
        help='directory for videos')
parser.add_argument(
        '--annotations_path',
        type=str,
        default="/research/rxtan/neurips_rebuttal/reuben/final_dataset_annotations.pkl",
        help='path to annotations')
parser.add_argument(
        '--segments_path',
        type=str,
        default="/research/rxtan/neurips_rebuttal/reuben/final_dataset_segments.pkl",
        help='path to video segments')
parser.add_argument(
        '--youcook2_trainval_annotations_path',
        type=str,
        default="/research/rxtan/neurips_rebuttal/pretrained_s3d_baseline/youcookii_annotations_trainval.json",
        help='path to original youcook2 annotations')
parser.add_argument(
        '--output_dir',
        type=str,
        default="./visualized_frames",
        help='directory to store annotated frames')
        
def get_video_frames(video_path):     
    cmd = (ffmpeg.probe(video_path))
    streams = cmd['streams'][0]
    width = streams['width']
    height = streams['height']
        
    cmd = (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=1)
    )
    out, _ = (
        cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return video
        
def main():
    args = parser.parse_args()
    
    #original_annotations = json.load(open(args.youcook2_trainval_annotations_path))['database']
    interactions_annotations = pickle.load(open(args.annotations_path, "rb"))
    interactions_segments = pickle.load(open(args.segments_path, "rb"))
    colors = (255, 0 , 0)
        
    for seg in (interactions_segments):
        video = seg[0]
        start = seg[1]
        end = seg[2]
        
        curr_orig_anns = original_annotations[video]['annotations']
        curr_loc_anns = interactions_annotations[video]
        
        if os.path.isfile(os.path.join(args.video_dir, video + '.mp4')):
            video_path = os.path.join(args.video_dir, video + '.mp4')
        elif os.path.isfile(os.path.join(args.video_dir, video + '.mkv')):
            video_path = os.path.join(args.video_dir, video + '.mkv')
        elif os.path.isfile(os.path.join(args.video_dir, video + '.webm')):
            video_path = os.path.join(args.video_dir, video + '.webm')
            
        video_frames = get_video_frames(video_path)
        
        '''text = None
        for tmp in curr_orig_anns:
            if tmp['segment'][0] == start:
                text = tmp['sentence']  
        print('text: ', text)'''
        
        for frame_id in curr_loc_anns:
            frame_ann = curr_loc_anns[frame_id]
            xtl = int(frame_ann[0])
            ytl = int(frame_ann[1])
            xbr = int(frame_ann[2])
            ybr = int(frame_ann[3])
            
            
            curr_frame = video_frames[frame_id-1]
            
            frame_dimensions = curr_frame.shape
            
            height = frame_dimensions[0]
            width = frame_dimensions[1]
            
            if height > 360 and width > 640:
                curr_frame = cv2.resize(curr_frame, (640, 360))
            
            curr_output_path = os.path.join(args.output_dir, '%s_____%s.png' % (video, frame_id))
            
            curr_frame = cv2.rectangle(curr_frame, (xtl, ytl), (xbr, ybr), colors, 2)
            cv2.imwrite(curr_output_path, cv2.cvtColor(curr_frame, cv2.COLOR_RGB2BGR))
  
if __name__ == "__main__":
    main()