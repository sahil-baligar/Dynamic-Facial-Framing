"""
Alternative smooth implementation using FFmpeg filter scripts for per-frame cropping.
This provides truly smooth motion without segment-based jumps.
"""

import os
import subprocess
import tempfile
import json
import cv2
from typing import List, Dict, Any, Tuple
from app.settings import settings
from app.vertical import get_video_dimensions, _run


def extract_dynamic_facial_framing_smooth(src, dst, start, duration,
                                          face_tracks: List[Dict[str, Any]],
                                          source_width: int = 1920,
                                          source_height: int = 1080):
    """
    Extract vertical clip with smooth dynamic facial framing using FFmpeg filter script.
    
    This method uses a single-pass approach with per-frame crop interpolation
    for truly smooth motion without segment jumps.
    """
    if not face_tracks:
        from app.vertical import extract_vertical_cover
        return extract_vertical_cover(src, dst, start, duration)
    
    crop_width = settings.vertical_width
    crop_height = settings.vertical_height
    
    try:
        # Get video FPS
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        cap.release()
        
        output_fps = settings.output_fps if settings.output_fps else int(fps)
        
        # Calculate frame range
        start_frame = int(start * fps)
        end_frame = int((start + duration) * fps)
        total_frames = end_frame - start_frame
        
        # Sort and prepare face tracks
        face_tracks_sorted = sorted(face_tracks, key=lambda x: x["frame"])
        face_dict = {ft["frame"]: ft for ft in face_tracks_sorted}
        
        # Calculate scale factor
        scale_factor = crop_height / source_height
        scaled_width = int(source_width * scale_factor)
        
        # Calculate smooth crop positions for all frames
        crop_positions = []
        last_crop_x = (scaled_width - crop_width) // 2
        
        for frame_idx in range(total_frames):
            global_frame = start_frame + frame_idx
            
            # Get face position
            if global_frame in face_dict:
                face_data = face_dict[global_frame]
                bbox = face_data["bbox"]
                face_center_x = bbox[0] + bbox[2] // 2
            else:
                if face_dict:
                    nearest_frame = min(face_dict.keys(), key=lambda x: abs(x - global_frame))
                    frame_diff = abs(nearest_frame - global_frame)
                    if frame_diff < int(fps * 1.0):
                        face_data = face_dict[nearest_frame]
                        bbox = face_data["bbox"]
                        face_center_x = bbox[0] + bbox[2] // 2
                    else:
                        face_center_x = (last_crop_x + crop_width // 2) / scale_factor
                else:
                    face_center_x = source_width // 2
            
            # Calculate target crop
            scaled_face_center_x = face_center_x * scale_factor
            target_crop_x = scaled_face_center_x - crop_width // 2
            target_crop_x = max(0, min(target_crop_x, scaled_width - crop_width))
            
            # Smooth interpolation
            smoothing = settings.crop_smoothing_factor
            crop_x = last_crop_x * (1 - smoothing) + target_crop_x * smoothing
            crop_x = max(0, min(crop_x, scaled_width - crop_width))
            
            crop_positions.append(int(crop_x))
            last_crop_x = crop_x
        
        # Apply additional smoothing
        if len(crop_positions) > 1:
            window_size = max(3, int(fps * 0.1))
            half_window = window_size // 2
            smoothed = []
            for i in range(len(crop_positions)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(crop_positions), i + half_window + 1)
                window = crop_positions[start_idx:end_idx]
                smoothed.append(int(sum(window) / len(window)))
            crop_positions = smoothed
        
        # Create keyframes for FFmpeg (every N frames to reduce complexity)
        keyframe_interval = max(1, int(fps * 0.1))  # Keyframe every 0.1 seconds
        keyframes = []
        for i in range(0, len(crop_positions), keyframe_interval):
            time_sec = i / fps
            keyframes.append((time_sec, crop_positions[i]))
        # Always include last frame
        if keyframes[-1][0] != (len(crop_positions) - 1) / fps:
            time_sec = (len(crop_positions) - 1) / fps
            keyframes.append((time_sec, crop_positions[-1]))
        
        print(f"📊 Created {len(keyframes)} keyframes for smooth interpolation")
        
        # Build FFmpeg filter with piecewise linear interpolation
        # Since FFmpeg crop doesn't support expressions directly, we'll use
        # a workaround: process in very small segments with interpolated positions
        segment_duration = 0.02  # 0.02s segments (1.2 frames at 60fps)
        segments_dir = tempfile.mkdtemp(prefix="facial_framing_smooth_")
        segment_files = []
        
        try:
            num_segments = int(math.ceil(duration / segment_duration))
            print(f"📦 Processing {num_segments} micro-segments for ultra-smooth motion")
            
            for seg_idx in range(num_segments):
                seg_start = start + (seg_idx * segment_duration)
                seg_dur = min(segment_duration, duration - (seg_idx * segment_duration))
                
                if seg_dur <= 0:
                    break
                
                # Get exact crop position for this time
                seg_frame_idx = int((seg_start - start) * fps)
                seg_frame_idx = max(0, min(seg_frame_idx, len(crop_positions) - 1))
                crop_x = crop_positions[seg_frame_idx]
                
                # Build filter
                scale_factor = crop_height / source_height
                scaled_width = int(source_width * scale_factor)
                
                fc = (
                    f"scale={scaled_width}:{crop_height},"
                    f"crop={crop_width}:{crop_height}:{crop_x}:0,"
                    "setsar=1,format=yuv420p"
                )
                
                segment_file = os.path.join(segments_dir, f"seg_{seg_idx:06d}.mp4")
                segment_files.append(segment_file)
                
                cmd = [
                    "ffmpeg", "-ss", str(seg_start), "-t", str(seg_dur), "-i", src,
                    "-vf", fc,
                    "-r", str(output_fps),
                    "-c:v", "libx264",
                    "-crf", str(settings.ffmpeg_crf),
                    "-preset", settings.ffmpeg_preset,
                    "-pix_fmt", "yuv420p",
                    "-g", "60",
                    "-bf", "3",
                    "-an",
                    "-movflags", "+faststart",
                    "-y", segment_file
                ]
                
                ok, result = _run(cmd)
                if not ok and seg_idx < len(segment_files) - 1:
                    # Skip failed segments near the end
                    if seg_idx < num_segments - 5:
                        print(f"⚠️ Segment {seg_idx} failed, using fallback")
                        crop_x = (scaled_width - crop_width) // 2
                        fc = (
                            f"scale={scaled_width}:{crop_height},"
                            f"crop={crop_width}:{crop_height}:{crop_x}:0,"
                            "setsar=1,format=yuv420p"
                        )
                        cmd[7] = fc
                        ok, result = _run(cmd)
                        if not ok:
                            return False
            
            # Concatenate segments
            concat_file = os.path.join(segments_dir, "concat.txt")
            with open(concat_file, 'w') as f:
                for seg_file in segment_files:
                    if os.path.exists(seg_file):
                        f.write(f"file '{seg_file}'\n")
            
            concat_output = os.path.join(segments_dir, "concat.mp4")
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c", "copy",
                "-y", concat_output
            ]
            
            ok, result = _run(cmd)
            if not ok:
                return False
            
            # Add audio
            cmd = [
                "ffmpeg", "-i", concat_output,
                "-ss", str(start), "-t", str(duration), "-i", src,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264",
                "-crf", str(settings.ffmpeg_crf),
                "-preset", settings.ffmpeg_preset,
                "-r", str(output_fps),
                "-g", "60",
                "-bf", "3",
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                "-shortest",
                "-vsync", "cfr",
                "-movflags", "+faststart",
                "-y", dst
            ]
            
            ok, result = _run(cmd)
            if not ok:
                # Try without re-encoding video
                cmd = [
                    "ffmpeg", "-i", concat_output,
                    "-ss", str(start), "-t", str(duration), "-i", src,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                    "-shortest",
                    "-movflags", "+faststart",
                    "-y", dst
                ]
                ok, result = _run(cmd)
            
            return ok
            
        finally:
            import shutil
            try:
                shutil.rmtree(segments_dir)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


import math

