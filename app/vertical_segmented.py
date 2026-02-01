"""
Segment-based dynamic layout switching for mixed content videos.
Processes video in segments and switches between modes (cover vs face tracking)
based on content in each segment.
"""

import os
import math
import subprocess
import tempfile
import shutil
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.vertical import (extract_vertical_cover, extract_dynamic_facial_framing, 
                          extract_podcast_background, extract_smaller_dynamic_framing,
                          extract_smaller_centered_frame,
                          get_video_dimensions, _run)
from app.layout.state_machine import LayoutStateMachine, LayoutState
from app.settings import settings


def extract_segmented_dynamic_layout(src, dst, start, duration, 
                                     state_machine: LayoutStateMachine,
                                     background_mode: str = "blur",
                                     segment_duration: float = 5.0,
                                     podcast_bg_path: Optional[str] = None,
                                     use_smaller_frame: bool = False) -> Dict[str, Any]:
    """
    Extract vertical clip with segment-based mode switching.
    
    Processes video in segments and switches between:
    - Simple cover crop (for gameplay/corner webcam segments)
    - Dynamic facial framing (for main subject face segments)
    - Podcast mode with background video (for high face detection segments)
    - Smaller frame mode with vertical tracking (optional)
    
    Args:
        src: Source video path
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        state_machine: Layout state machine instance
        background_mode: "blur", "gameplay", or "podcast"
        segment_duration: Duration of each analysis segment (seconds)
        podcast_bg_path: Path to podcast background video (optional)
        use_smaller_frame: If True, use smaller frame mode with vertical tracking
        
    Returns:
        Dict with success status and segment information
    """
    print(f"🎬 Segment-based dynamic layout extraction: {start:.2f}s to {start + duration:.2f}s")
    print(f"📦 Analyzing in {segment_duration}s segments for mode switching")
    
    # Get video info
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    source_width, source_height = get_video_dimensions(src)
    output_fps = settings.output_fps if settings.output_fps else int(fps)
    
    # Calculate number of segments
    num_segments = int(math.ceil(duration / segment_duration))
    segments_dir = tempfile.mkdtemp(prefix="facial_framing_segments_")
    segment_files = []
    segment_modes = []
    
    try:
        print(f"📊 Processing {num_segments} segments...")
        
        for seg_idx in range(num_segments):
            seg_start = start + (seg_idx * segment_duration)
            seg_dur = min(segment_duration, duration - (seg_idx * segment_duration))
            
            if seg_dur <= 0:
                break
            
            print(f"\n🔄 Segment {seg_idx + 1}/{num_segments}: {seg_start:.1f}s - {seg_start + seg_dur:.1f}s")
            
            # Analyze this segment to determine mode
            seg_state_machine = LayoutStateMachine()
            seg_face_tracks = []
            
            # Analyze frames in this segment
            # Sample frames at regular intervals for faster analysis (every Nth frame)
            cap = cv2.VideoCapture(src)
            seg_start_frame = int(seg_start * fps)
            seg_end_frame = int((seg_start + seg_dur) * fps)
            
            frame_height, frame_width = None, None
            
            # Sample every 5th frame for faster analysis (still accurate enough)
            frame_step = max(1, int(fps * 0.1))  # Sample every ~0.1 seconds
            
            for frame_num in range(seg_start_frame, min(seg_end_frame, total_frames), frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_height is None:
                    frame_height, frame_width = frame.shape[:2]
                
                # Update state machine
                state = seg_state_machine.update_state(frame, frame_num, background_mode)
                layout_config = seg_state_machine.get_current_layout_config()
                
                # Collect main subject faces only
                if layout_config.get("primary_face"):
                    face_track = layout_config["primary_face"]
                    bbox = face_track.get("bbox", [0, 0, 0, 0])
                    
                    # Verify this is a main subject face
                    face_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
                    frame_area = frame_width * frame_height
                    face_area_percentage = (face_area / frame_area * 100) if frame_area > 0 else 0
                    
                    # For smaller frame mode, be more lenient with face detection
                    min_area_threshold = settings.min_face_area_percentage
                    if use_smaller_frame and settings.smaller_frame_enabled:
                        min_area_threshold = max(1.0, min_area_threshold * 0.5)  # Lower threshold for smaller frame mode
                    
                    if face_area_percentage >= min_area_threshold:
                        face_center_x = bbox[0] + bbox[2] / 2
                        face_center_y = bbox[1] + bbox[3] / 2
                        
                        corner_threshold = settings.corner_webcam_threshold
                        is_in_corner = (
                            (face_center_x < frame_width * corner_threshold or 
                             face_center_x > frame_width * (1 - corner_threshold)) and
                            (face_center_y < frame_height * corner_threshold or 
                             face_center_y > frame_height * (1 - corner_threshold))
                        )
                        
                        if not is_in_corner:
                            seg_face_tracks.append({
                                "frame": frame_num,
                                "bbox": bbox,
                                "confidence": face_track.get("confidence", 0.5)
                            })
            
            cap.release()
            
            # If we sampled frames, interpolate face tracks for intermediate frames
            # This ensures we have face data for all frames, not just sampled ones
            if frame_step > 1 and seg_face_tracks:
                # Sort by frame number
                seg_face_tracks.sort(key=lambda x: x["frame"])
                
                # Fill in gaps by interpolating between detected frames
                filled_tracks = []
                last_face = None
                
                for frame_num in range(seg_start_frame, min(seg_end_frame, total_frames)):
                    # Find nearest detected face
                    nearest_face = None
                    min_diff = float('inf')
                    
                    for face_track in seg_face_tracks:
                        diff = abs(face_track["frame"] - frame_num)
                        if diff < min_diff:
                            min_diff = diff
                            nearest_face = face_track
                    
                    # Use nearest face if within reasonable distance (0.5 seconds)
                    if nearest_face and min_diff < int(fps * 0.5):
                        filled_tracks.append({
                            "frame": frame_num,
                            "bbox": nearest_face["bbox"].copy(),
                            "confidence": nearest_face["confidence"]
                        })
                    elif last_face:
                        # Use last face if no nearby face found
                        filled_tracks.append({
                            "frame": frame_num,
                            "bbox": last_face["bbox"].copy(),
                            "confidence": last_face["confidence"]
                        })
                    elif nearest_face:
                        # Use nearest face even if far away (better than nothing)
                        filled_tracks.append({
                            "frame": frame_num,
                            "bbox": nearest_face["bbox"].copy(),
                            "confidence": nearest_face["confidence"]
                        })
                    
                    # Update last_face if we found one
                    if nearest_face and min_diff < int(fps * 0.5):
                        last_face = nearest_face
                
                seg_face_tracks = filled_tracks
            
            # Determine mode for this segment
            seg_total_frames = seg_end_frame - seg_start_frame
            seg_face_percentage = (len(seg_face_tracks) / seg_total_frames * 100) if seg_total_frames > 0 else 0
            has_main_subject = (seg_face_percentage >= settings.min_face_detection_percentage and 
                              len(seg_face_tracks) >= settings.min_face_detections)
            is_podcast = seg_face_percentage >= settings.podcast_face_detection_threshold
            
            # Create segment output file
            seg_output = os.path.join(segments_dir, f"segment_{seg_idx:04d}.mp4")
            segment_files.append(seg_output)
            
            # Check smaller frame mode FIRST (priority when enabled)
            if use_smaller_frame and settings.smaller_frame_enabled:
                # Smaller frame mode is enabled - use it even if faces aren't detected
                # If faces detected, use dynamic tracking; otherwise, use centered smaller frame
                if has_main_subject and seg_face_tracks:
                    print(f"  📐 Main subject detected ({seg_face_percentage:.1f}%) - Using smaller dynamic framing with vertical + horizontal tracking")
                    segment_modes.append("smaller_frame")
                    success = extract_smaller_dynamic_framing(
                        src, seg_output, seg_start, seg_dur,
                        seg_face_tracks, source_width, source_height
                    )
                else:
                    # No faces detected but smaller frame mode enabled - use centered smaller frame
                    print(f"  📐 No faces detected ({seg_face_percentage:.1f}%) - Using centered smaller frame")
                    segment_modes.append("smaller_frame")
                    success = extract_smaller_centered_frame(
                        src, seg_output, seg_start, seg_dur,
                        source_width, source_height
                    )
            elif is_podcast and podcast_bg_path and os.path.exists(podcast_bg_path):
                # Podcast detected - use podcast background mode with fast teleport
                print(f"  🎙️ Podcast detected ({seg_face_percentage:.1f}%) - Using podcast background with fast teleport")
                segment_modes.append("podcast")
                success = extract_podcast_background(
                    src, seg_output, seg_start, seg_dur, podcast_bg_path,
                    seg_face_tracks, source_width, source_height,
                    use_fast_teleport=True
                )
            elif has_main_subject:
                # Main subject face detected - use dynamic facial framing
                print(f"  👤 Main subject detected ({seg_face_percentage:.1f}%) - Using dynamic facial framing")
                segment_modes.append("face_tracking")
                success = extract_dynamic_facial_framing(
                    src, seg_output, seg_start, seg_dur,
                    seg_face_tracks, source_width, source_height
                )
            else:
                # No main subject - use simple cover crop
                print(f"  🎮 No main subject ({seg_face_percentage:.1f}%) - Using simple cover crop")
                segment_modes.append("cover")
                success = extract_vertical_cover(src, seg_output, seg_start, seg_dur)
            
            if not success:
                print(f"  ⚠️ Segment {seg_idx} processing failed")
                return {"success": False, "error": f"Segment {seg_idx} processing failed"}
        
        # Concatenate segments
        print(f"\n🔗 Concatenating {len(segment_files)} segments...")
        concat_file = os.path.join(segments_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for seg_file in segment_files:
                if os.path.exists(seg_file):
                    f.write(f"file '{seg_file}'\n")
        
        concat_output = os.path.join(segments_dir, "concat_output.mp4")
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy",
            "-y", concat_output
        ]
        
        ok, result = _run(cmd)
        if not ok:
            print(f"❌ Concatenation failed: {result.stderr[:500]}")
            return {"success": False, "error": "Concatenation failed"}
        
        # Copy final output
        shutil.copy2(concat_output, dst)
        
        print(f"\n✅ Successfully processed {num_segments} segments")
        print(f"📊 Mode distribution:")
        cover_count = segment_modes.count("cover")
        face_count = segment_modes.count("face_tracking")
        podcast_count = segment_modes.count("podcast")
        smaller_frame_count = segment_modes.count("smaller_frame")
        print(f"   Cover crop: {cover_count} segments")
        print(f"   Face tracking: {face_count} segments")
        print(f"   Podcast: {podcast_count} segments")
        print(f"   Smaller frame: {smaller_frame_count} segments")
        
        return {
            "success": True,
            "num_segments": num_segments,
            "segment_modes": segment_modes,
            "cover_segments": cover_count,
            "face_tracking_segments": face_count,
            "podcast_segments": podcast_count,
            "smaller_frame_segments": smaller_frame_count
        }
        
    except Exception as e:
        print(f"❌ Error in segment-based processing: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(segments_dir)
        except Exception as e:
            print(f"⚠️ Failed to cleanup: {e}")

