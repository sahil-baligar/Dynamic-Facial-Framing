"""
Vertical 9:16 video rendering for ClipGenius Pipeline v2.

Integrated with dynamic layout state machine for face-aware rendering.
"""

import os
import subprocess
import random
import json
import math
import tempfile
import cv2  # pip install opencv-python
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from app.settings import settings
from app.layout.state_machine import LayoutStateMachine, LayoutState, FaceTrack

# ---- helpers ---------------------------------------------------------------

def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r

def _ffmpeg_common_outargs():
    return [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-r", "30", "-fps_mode", "cfr",
        "-profile:v", "baseline", "-level:v", "3.0", "-tag:v", "avc1",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart", "-y"
    ]

# ---- LAYOUT 1: full cover crop (no distortion) ----------------------------

def extract_vertical_cover(src, dst, start, duration):
    """
    Extract vertical 9:16 clip using simple cover crop.
    This is the original horizontal-to-vertical conversion - used when no human/speaker is detected.
    Perfect for gameplay videos, landscape footage, or any non-human content.
    """
    # Get video FPS to preserve frame rate
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()
    
    output_fps = settings.output_fps if settings.output_fps else int(fps)
    crop_width = settings.vertical_width
    crop_height = settings.vertical_height
    
    # Rock-solid filter: scale to cover, crop exactly, force SAR=1, yuv420p
    fc = (
        f"scale={crop_width}:{crop_height}:force_original_aspect_ratio=increase,"
        f"crop={crop_width}:{crop_height},"
        "setsar=1,"
        "format=yuv420p"
    )
    
    cmd = [
        "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
        "-vf", fc,
        "-r", str(output_fps),
        "-c:v", "libx264",
        "-crf", str(settings.ffmpeg_crf),
        "-preset", settings.ffmpeg_preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        "-y", dst
    ]
    
    print(f"🎬 Running vertical cover command: {' '.join(cmd)}")
    ok, result = _run(cmd)
    
    if not ok:
        print(f"❌ Vertical cover extraction failed: {result.stderr}")
        # Log first 30 lines of stderr for debugging
        stderr_lines = result.stderr.split('\n')[:30]
        print(f"🔍 First 30 lines of stderr:")
        for line in stderr_lines:
            print(f"   {line}")
        return False
    
    # Verify output dimensions
    if os.path.exists(dst):
        try:
            probe_cmd = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,sample_aspect_ratio",
                "-of", "csv=p=0", dst
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                output = probe_result.stdout.strip()
                print(f"✅ Vertical cover output verified: {output}")
                return True
            else:
                print(f"⚠️ Could not verify output dimensions: {probe_result.stderr}")
                return True  # Assume success if we can't verify
        except Exception as e:
            print(f"⚠️ Verification failed: {e}")
            return True  # Assume success if verification fails
    
    return False

# ---- LAYOUT 2: podcast face-centered crop ---------------------------------

def extract_podcast_face(src, dst, start, duration, primary_face: Optional[FaceTrack] = None):
    """Extract vertical clip with face-centered crop for podcast content"""
    if primary_face and primary_face.bbox:
        # Use detected face for crop center
        x, y, w, h = primary_face.bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Ensure crop is within bounds and maintains 9:16 aspect ratio
        crop_width = 1080
        crop_height = 1920
        
        # Calculate crop coordinates ensuring we stay within frame bounds
        crop_x = max(0, min(center_x - crop_width // 2, 1920 - crop_width))
        crop_y = max(0, min(center_y - crop_height // 2, 1080 - crop_height))
        
        # Use crop filter for precise face-centered extraction
        fc = (
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
            "scale=1080:1920,setsar=1,format=yuv420p"
        )
    else:
        # Fallback to center crop if no face detected
        fc = (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,setsar=1,format=yuv420p"
        )
    
    cmd = [
        "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
        "-vf", fc,
        "-r", "30",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", dst
    ]
    
    print(f"🎬 Running podcast face command: {' '.join(cmd)}")
    ok, result = _run(cmd)
    
    if not ok:
        print(f"❌ Podcast face extraction failed: {result.stderr}")
        return False
    
    return True

# ---- LAYOUT 3: blurred background with foreground rectangle ----------------

def extract_blur_background(src, dst, start, duration):
    """Extract clip with blurred background and foreground rectangle overlay"""
    fc = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,boxblur=40:20,crop=1080:1920,setsar=1[bg];"
        "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[fgr];"
        "[bg][fgr]overlay=0:0,format=yuv420p"
    )
    
    cmd = [
        "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
        "-filter_complex", fc,
        "-r", "30",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y", dst
    ]
    
    print(f"🎬 Running blur background command: {' '.join(cmd)}")
    ok, result = _run(cmd)
    
    if not ok:
        print(f"❌ Blur background extraction failed: {result.stderr}")
        stderr_lines = result.stderr.split('\n')[:30]
        print(f"🔍 First 30 lines of stderr:")
        for line in stderr_lines:
            print(f"   {line}")
        return False
    
    return True

# ---- LAYOUT 4: gameplay background with looping video --------------------

def extract_gameplay_background(src, dst, start, duration, game_bg_path):
    """Extract clip with looping gameplay background and foreground overlay"""
    if not os.path.exists(game_bg_path):
        print(f"⚠️ Game background not found: {game_bg_path}")
        return extract_vertical_cover(src, dst, start, duration)
    
    fc = (
        "[1:v]scale=1080:1920:force_original_aspect_ratio=cover,setsar=1[game];"
        "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1[fgr];"
        "[game][fgr]overlay=0:0,format=yuv420p"
    )
    
    cmd = [
        "ffmpeg", "-stream_loop", "-1", "-i", game_bg_path,
        "-ss", str(start), "-t", str(duration), "-i", src,
        "-filter_complex", fc,
        "-shortest",
        "-r", "30",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-y", dst
    ]
    
    print(f"🎮 Running gameplay background command: {' '.join(cmd)}")
    ok, result = _run(cmd)
    
    if not ok:
        print(f"❌ Gameplay background extraction failed: {result.stderr}")
        stderr_lines = result.stderr.split('\n')[:30]
        print(f"🔍 First 30 lines of stderr:")
        for line in stderr_lines:
            print(f"   {line}")
        return False
    
    return True

# ---- LAYOUT 5: Podcast background with split screen --------------------

def extract_podcast_background(src, dst, start, duration, bg_video_path, 
                                face_tracks: List[Dict[str, Any]] = None,
                                source_width: int = 1920, 
                                source_height: int = 1080,
                                use_fast_teleport: bool = True):
    """
    Extract podcast clip with background video split screen.
    Left half: podcast speakers (with facial tracking)
    Right half: background video (subway surfers gameplay)
    
    Args:
        src: Source video path (podcast)
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        bg_video_path: Path to background video
        face_tracks: List of face track dictionaries (optional, for fast teleport)
        source_width: Source video width
        source_height: Source video height
        use_fast_teleport: If True, use fast teleport to speaker (instant jump)
    """
    if not os.path.exists(bg_video_path):
        print(f"⚠️ Background video not found: {bg_video_path}")
        print(f"📐 Falling back to simple vertical cover")
        return extract_vertical_cover(src, dst, start, duration)
    
    # Output dimensions
    output_width = settings.vertical_width
    output_height = settings.vertical_height
    
    # Split screen: left half podcast, right half background
    podcast_width = output_width // 2  # Left half
    bg_width = output_width - podcast_width  # Right half
    
    # Get video FPS
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()
    
    output_fps = settings.output_fps if settings.output_fps else int(fps)
    
    # If face tracks provided and fast teleport enabled, use dynamic framing
    if face_tracks and use_fast_teleport and settings.fast_teleport_enabled:
        print(f"🎙️ Podcast mode: Fast teleport facial framing with background video")
        
        # Create temporary file for podcast side with facial tracking
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="podcast_temp_")
        podcast_side = os.path.join(temp_dir, "podcast_side.mp4")
        
        try:
            # Use fast teleport facial framing for podcast side
            success = extract_fast_teleport_facial_framing(
                src, podcast_side, start, duration,
                face_tracks, source_width, source_height,
                crop_width=podcast_width, crop_height=output_height
            )
            
            if not success:
                print(f"⚠️ Fast teleport failed, using simple crop")
                # Fallback to simple crop
                crop_cmd = [
                    "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
                    "-vf", f"scale={podcast_width}:{output_height}:force_original_aspect_ratio=increase,crop={podcast_width}:{output_height},setsar=1,format=yuv420p",
                    "-r", str(output_fps),
                    "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
                    "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart", "-y", podcast_side
                ]
                ok, result = _run(crop_cmd)
                if not ok:
                    print(f"❌ Podcast side extraction failed: {result.stderr[:500]}")
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return False
        except Exception as e:
            print(f"⚠️ Error in fast teleport: {e}, using simple crop")
            # Fallback to simple crop
            crop_cmd = [
                "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
                "-vf", f"scale={podcast_width}:{output_height}:force_original_aspect_ratio=increase,crop={podcast_width}:{output_height},setsar=1,format=yuv420p",
                "-r", str(output_fps),
                "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
                "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
                "-an", "-movflags", "+faststart", "-y", podcast_side
            ]
            ok, result = _run(crop_cmd)
            if not ok:
                print(f"❌ Podcast side extraction failed: {result.stderr[:500]}")
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
    else:
        # Simple crop for podcast side
        print(f"🎙️ Podcast mode: Simple crop with background video")
        temp_dir = tempfile.mkdtemp(prefix="podcast_temp_")
        podcast_side = os.path.join(temp_dir, "podcast_side.mp4")
        crop_cmd = [
            "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
            "-vf", f"scale={podcast_width}:{output_height}:force_original_aspect_ratio=increase,crop={podcast_width}:{output_height},setsar=1,format=yuv420p",
            "-r", str(output_fps),
            "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
            "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", "-y", podcast_side
        ]
        ok, result = _run(crop_cmd)
        if not ok:
            print(f"❌ Podcast side extraction failed: {result.stderr[:500]}")
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    # Scale background video to fit right half
    # Background video should fill the right half (bg_width x output_height)
    fc = (
        f"[0:v]scale={podcast_width}:{output_height}:force_original_aspect_ratio=increase,"
        f"crop={podcast_width}:{output_height},setsar=1[podcast];"
        f"[1:v]scale={bg_width}:{output_height}:force_original_aspect_ratio=cover,setsar=1[bg];"
        f"[podcast][bg]hstack=inputs=2,format=yuv420p"
    )
    
    cmd = [
        "ffmpeg", "-i", podcast_side,
        "-stream_loop", "-1", "-i", bg_video_path,
        "-filter_complex", fc,
        "-shortest",
        "-r", str(output_fps),
        "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
        "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-y", dst
    ]
    
    print(f"🎙️ Running podcast background command: {' '.join(cmd)}")
    ok, result = _run(cmd)
    
    # Cleanup
    try:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠️ Failed to cleanup temp files: {e}")
    
    if not ok:
        print(f"❌ Podcast background extraction failed: {result.stderr[:500]}")
        return False
    
    return True

# ---- Fast teleport facial framing (for podcasts) --------------------------

def extract_fast_teleport_facial_framing(src, dst, start, duration,
                                         face_tracks: List[Dict[str, Any]],
                                         source_width: int = 1920,
                                         source_height: int = 1080,
                                         crop_width: int = None,
                                         crop_height: int = None):
    """
    Extract vertical clip with fast teleport facial framing.
    Teleports instantly to speaker (no gradual movement), then tracks smoothly.
    
    Args:
        src: Source video path
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        face_tracks: List of face track dictionaries
        source_width: Source video width
        source_height: Source video height
        crop_width: Crop width (default: settings.vertical_width)
        crop_height: Crop height (default: settings.vertical_height)
    """
    if not face_tracks:
        print("⚠️ No face tracks provided, falling back to center crop")
        return extract_vertical_cover(src, dst, start, duration)
    
    # Use provided crop dimensions or defaults
    if crop_width is None:
        crop_width = settings.vertical_width
    if crop_height is None:
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
        
        # Sort face tracks by frame number
        face_tracks_sorted = sorted(face_tracks, key=lambda x: x["frame"])
        face_dict = {ft["frame"]: ft for ft in face_tracks_sorted}
        
        # Calculate scale factor
        scale_factor = crop_height / source_height
        scaled_width = int(source_width * scale_factor)
        scaled_height = int(source_height * scale_factor)
        
        # Fast teleport: instant jump to speaker, then smooth tracking
        crop_positions = []
        current_crop_x = (scaled_width - crop_width) // 2  # Start centered
        current_crop_y = 0
        last_face_frame = None
        teleport_count = 0
        
        for frame_idx in range(total_frames):
            global_frame = start_frame + frame_idx
            
            # Get face position for this frame
            if global_frame in face_dict:
                face_data = face_dict[global_frame]
                bbox = face_data["bbox"]
                face_center_x = bbox[0] + bbox[2] // 2
                face_center_y = bbox[1] + bbox[3] // 2
                
                # Convert to scaled coordinates
                scaled_face_center_x = face_center_x * scale_factor
                scaled_face_center_y = face_center_y * scale_factor
                
                # Calculate target crop position (center face in crop)
                target_crop_x = scaled_face_center_x - crop_width // 2
                # For vertical movement: center face vertically in crop
                # Only use vertical movement if crop_height < scaled_height (room to move)
                if crop_height < scaled_height:
                    target_crop_y = scaled_face_center_y - crop_height // 2
                else:
                    target_crop_y = 0  # No vertical movement if crop fills height
                
                # Apply bounds
                target_crop_x = max(0, min(target_crop_x, scaled_width - crop_width))
                target_crop_y = max(0, min(target_crop_y, scaled_height - crop_height))
                
                # Fast teleport: if face moved significantly or new face, teleport instantly
                if last_face_frame is None:
                    # First face detection - teleport instantly
                    current_crop_x = target_crop_x
                    current_crop_y = target_crop_y
                    teleport_count += 1
                else:
                    # Check if we should teleport (large movement or significant time gap)
                    frame_diff = global_frame - last_face_frame
                    distance = ((target_crop_x - current_crop_x) ** 2 + (target_crop_y - current_crop_y) ** 2) ** 0.5
                    
                    # Teleport if: large movement (>20% of crop width) OR significant time gap (>threshold frames)
                    should_teleport = (distance > crop_width * 0.2) or (frame_diff > settings.teleport_threshold_frames)
                    
                    if should_teleport:
                        # Instant teleport
                        current_crop_x = target_crop_x
                        current_crop_y = target_crop_y
                        teleport_count += 1
                    else:
                        # Smooth tracking after teleport
                        smoothing = settings.smaller_frame_smoothing if settings.smaller_frame_enabled else settings.crop_smoothing_factor
                        current_crop_x = int(current_crop_x * (1 - smoothing) + target_crop_x * smoothing)
                        current_crop_y = int(current_crop_y * (1 - smoothing) + target_crop_y * smoothing)
                
                last_face_frame = global_frame
            else:
                # No face in this frame - find nearest
                if face_dict:
                    nearest_frame = min(face_dict.keys(), key=lambda x: abs(x - global_frame))
                    frame_diff = abs(nearest_frame - global_frame)
                    max_interpolation_frames = int(fps * 0.5)  # Shorter interpolation window
                    
                    if frame_diff < max_interpolation_frames:
                        face_data = face_dict[nearest_frame]
                        bbox = face_data["bbox"]
                        face_center_x = bbox[0] + bbox[2] // 2
                        face_center_y = bbox[1] + bbox[3] // 2
                        
                        scaled_face_center_x = face_center_x * scale_factor
                        scaled_face_center_y = face_center_y * scale_factor
                        
                        target_crop_x = scaled_face_center_x - crop_width // 2
                        # Use vertical movement if room available
                        if crop_height < scaled_height:
                            target_crop_y = scaled_face_center_y - crop_height // 2
                        else:
                            target_crop_y = 0
                        
                        target_crop_x = max(0, min(target_crop_x, scaled_width - crop_width))
                        target_crop_y = max(0, min(target_crop_y, scaled_height - crop_height))
                        
                        # Smooth interpolation when no face detected
                        smoothing = settings.smaller_frame_smoothing if settings.smaller_frame_enabled else settings.crop_smoothing_factor
                        current_crop_x = int(current_crop_x * (1 - smoothing) + target_crop_x * smoothing)
                        current_crop_y = int(current_crop_y * (1 - smoothing) + target_crop_y * smoothing)
            
            crop_positions.append((int(current_crop_x), int(current_crop_y)))
        
        print(f"⚡ Fast teleport facial framing: {teleport_count} teleports, {len(face_tracks)} face detections")
        print(f"📊 Crop positions calculated: {len(crop_positions)} frames")
        
        # Process in segments
        import tempfile
        segments_dir = tempfile.mkdtemp(prefix="fast_teleport_segments_")
        segment_files = []
        num_segments = int(math.ceil(duration / settings.segment_duration))
        
        for seg_idx in range(num_segments):
            seg_start = start + (seg_idx * settings.segment_duration)
            seg_dur = min(settings.segment_duration, duration - (seg_idx * settings.segment_duration))
            
            if seg_dur <= 0:
                break
            
            seg_start_frame = int(seg_start * fps)
            seg_frame_idx = seg_start_frame - start_frame
            
            if seg_frame_idx < 0 or seg_frame_idx >= len(crop_positions):
                continue
            
            crop_x, crop_y = crop_positions[seg_frame_idx]
            
            # Create segment
            segment_file = os.path.join(segments_dir, f"segment_{seg_idx:04d}.mp4")
            segment_files.append(segment_file)
            
            fc = (
                f"scale={scaled_width}:{scaled_height},"
                f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
                "setsar=1,format=yuv420p"
            )
            
            cmd = [
                "ffmpeg", "-ss", str(seg_start), "-t", str(seg_dur), "-i", src,
                "-vf", fc,
                "-r", str(output_fps),
                "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
                "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
                "-g", str(int(output_fps * 2)),
                "-bf", "3",
                "-an",
                "-movflags", "+faststart",
                "-y", segment_file
            ]
            
            ok, result = _run(cmd)
            if not ok:
                print(f"⚠️ Segment {seg_idx} failed: {result.stderr[:200]}")
                # Use fallback
                fallback_cmd = [
                    "ffmpeg", "-ss", str(seg_start), "-t", str(seg_dur), "-i", src,
                    "-vf", f"scale={crop_width}:{crop_height}:force_original_aspect_ratio=increase,crop={crop_width}:{crop_height},setsar=1,format=yuv420p",
                    "-r", str(output_fps),
                    "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
                    "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart", "-y", segment_file
                ]
                ok, result = _run(fallback_cmd)
        
        # Concatenate segments
        concat_file = os.path.join(segments_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for seg_file in segment_files:
                if os.path.exists(seg_file):
                    f.write(f"file '{seg_file}'\n")
        
        concat_output = os.path.join(segments_dir, "concat_output.mp4")
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-y", concat_output
        ]
        
        ok, result = _run(cmd)
        if not ok:
            print(f"❌ Concatenation failed: {result.stderr[:500]}")
            import shutil
            shutil.rmtree(segments_dir)
            return False
        
        # Copy final output
        import shutil
        shutil.copy2(concat_output, dst)
        shutil.rmtree(segments_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Fast teleport facial framing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ---- Smaller dynamic frame with vertical tracking --------------------------

def extract_smaller_dynamic_framing(src, dst, start, duration,
                                   face_tracks: List[Dict[str, Any]],
                                   source_width: int = 1920,
                                   source_height: int = 1080):
    """
    Extract smaller vertical clip with dynamic facial framing that moves in all directions.
    Frame is smaller (80% size) but can track vertically and horizontally.
    
    Args:
        src: Source video path
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        face_tracks: List of face track dictionaries
        source_width: Source video width
        source_height: Source video height
    """
    if not face_tracks:
        print("⚠️ No face tracks provided, falling back to center crop")
        return extract_vertical_cover(src, dst, start, duration)
    
    # Smaller frame dimensions
    crop_width = settings.smaller_frame_width
    crop_height = settings.smaller_frame_height
    
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
        
        # Sort face tracks
        face_tracks_sorted = sorted(face_tracks, key=lambda x: x["frame"])
        face_dict = {ft["frame"]: ft for ft in face_tracks_sorted}
        
        # Calculate scale factor - scale LARGER than crop to allow vertical movement
        # We need extra vertical space (20% more) to enable up/down tracking
        vertical_padding_factor = 1.2  # 20% extra vertical space for movement
        scale_factor = (crop_height * vertical_padding_factor) / source_height
        scaled_width = int(source_width * scale_factor)
        scaled_height = int(source_height * scale_factor)
        
        # Ensure we have room for vertical movement
        if scaled_height <= crop_height:
            # If still no room, scale even larger
            scale_factor = (crop_height * 1.5) / source_height
            scaled_width = int(source_width * scale_factor)
            scaled_height = int(source_height * scale_factor)
        
        print(f"📐 Scale factor: {scale_factor:.3f}, Scaled: {scaled_width}x{scaled_height}, Crop: {crop_width}x{crop_height}")
        print(f"📐 Vertical room: {scaled_height - crop_height}px (allows {scaled_height - crop_height}px vertical movement)")
        
        # Dynamic tracking with vertical movement
        crop_positions = []
        last_crop_x = (scaled_width - crop_width) // 2
        last_crop_y = (scaled_height - crop_height) // 2  # Start centered vertically
        
        for frame_idx in range(total_frames):
            global_frame = start_frame + frame_idx
            
            # Get face position
            if global_frame in face_dict:
                face_data = face_dict[global_frame]
                bbox = face_data["bbox"]
                face_center_x = bbox[0] + bbox[2] // 2
                face_center_y = bbox[1] + bbox[3] // 2
            else:
                # Find nearest face
                if face_dict:
                    nearest_frame = min(face_dict.keys(), key=lambda x: abs(x - global_frame))
                    frame_diff = abs(nearest_frame - global_frame)
                    max_interpolation_frames = int(fps * 1.0)
                    
                    if frame_diff < max_interpolation_frames:
                        face_data = face_dict[nearest_frame]
                        bbox = face_data["bbox"]
                        face_center_x = bbox[0] + bbox[2] // 2
                        face_center_y = bbox[1] + bbox[3] // 2
                    else:
                        # No face nearby - center on last position
                        face_center_x = (last_crop_x + crop_width // 2) / scale_factor
                        face_center_y = (last_crop_y + crop_height // 2) / scale_factor
                else:
                    face_center_x = source_width // 2
                    face_center_y = source_height // 2
            
            # Convert to scaled coordinates
            scaled_face_center_x = face_center_x * scale_factor
            scaled_face_center_y = face_center_y * scale_factor
            
            # Calculate target crop position (center face in crop)
            target_crop_x = scaled_face_center_x - crop_width // 2
            target_crop_y = scaled_face_center_y - crop_height // 2
            
            # Apply bounds
            target_crop_x = max(0, min(target_crop_x, scaled_width - crop_width))
            target_crop_y = max(0, min(target_crop_y, scaled_height - crop_height))
            
            # Smooth interpolation (faster smoothing for smaller frame)
            smoothing = settings.smaller_frame_smoothing
            crop_x = int(last_crop_x * (1 - smoothing) + target_crop_x * smoothing)
            crop_y = int(last_crop_y * (1 - smoothing) + target_crop_y * smoothing)
            
            # Ensure bounds
            crop_x = max(0, min(crop_x, scaled_width - crop_width))
            crop_y = max(0, min(crop_y, scaled_height - crop_height))
            
            crop_positions.append((crop_x, crop_y))
            last_crop_x = crop_x
            last_crop_y = crop_y
        
        # Apply additional smoothing pass
        if len(crop_positions) > 1:
            smoothed_positions = []
            window_size = max(3, int(fps * 0.08))
            half_window = window_size // 2
            
            for i in range(len(crop_positions)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(crop_positions), i + half_window + 1)
                window = crop_positions[start_idx:end_idx]
                
                avg_x = sum(x for x, y in window) / len(window)
                avg_y = sum(y for x, y in window) / len(window)
                
                smoothed_positions.append((int(avg_x), int(avg_y)))
            
            crop_positions = smoothed_positions
            print(f"✨ Applied smoothing filter (window: {window_size} frames)")
        
        print(f"📐 Smaller dynamic framing: {crop_width}x{crop_height} (80% size)")
        print(f"📊 Crop positions calculated: {len(crop_positions)} frames")
        print(f"🎯 Vertical + horizontal tracking enabled")
        
        # Process in segments with per-frame crop positions
        # Use smaller segments (every 0.1 seconds) for smooth movement
        import tempfile
        segments_dir = tempfile.mkdtemp(prefix="smaller_frame_segments_")
        segment_files = []
        # Use smaller segment duration for smoother movement (0.1 seconds = ~6 frames at 60fps)
        segment_dur = 0.1
        num_segments = int(math.ceil(duration / segment_dur))
        
        # Calculate crop position range for debugging
        if crop_positions:
            crop_x_values = [x for x, y in crop_positions]
            crop_y_values = [y for x, y in crop_positions]
            min_x, max_x = min(crop_x_values), max(crop_x_values)
            min_y, max_y = min(crop_y_values), max(crop_y_values)
            print(f"📊 Crop position range: X=[{min_x}, {max_x}] (Δ{max_x-min_x}px), Y=[{min_y}, {max_y}] (Δ{max_y-min_y}px)")
        
        for seg_idx in range(num_segments):
            seg_start = start + (seg_idx * segment_dur)
            seg_dur_actual = min(segment_dur, duration - (seg_idx * segment_dur))
            
            if seg_dur_actual <= 0:
                break
            
            # Calculate frame indices for this segment
            seg_start_frame_local = int((seg_start - start) * fps)
            seg_end_frame_local = int((seg_start + seg_dur_actual - start) * fps)
            
            # Get crop positions for all frames in this segment
            segment_crop_positions = []
            for local_frame_idx in range(seg_start_frame_local, min(seg_end_frame_local, len(crop_positions))):
                if local_frame_idx >= 0 and local_frame_idx < len(crop_positions):
                    segment_crop_positions.append(crop_positions[local_frame_idx])
            
            if not segment_crop_positions:
                # Fallback: use center crop
                crop_x = (scaled_width - crop_width) // 2
                crop_y = (scaled_height - crop_height) // 2
            else:
                # For smooth movement, use the middle crop position of the segment
                # This ensures movement is visible within the segment
                mid_idx = len(segment_crop_positions) // 2
                crop_x, crop_y = segment_crop_positions[mid_idx]
            
            # Create segment with dynamic crop
            # For very short segments (0.1s), we use a single crop position
            # For smoother movement, we could interpolate, but FFmpeg's crop filter
            # doesn't easily support per-frame expressions without complex filter chains
            segment_file = os.path.join(segments_dir, f"segment_{seg_idx:04d}.mp4")
            segment_files.append(segment_file)
            
            # Use crop with the calculated position
            # Note: For true per-frame movement, we'd need to use FFmpeg expressions
            # or process frame-by-frame, which is slower but smoother
            fc = (
                f"scale={scaled_width}:{scaled_height},"
                f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
                "setsar=1,format=yuv420p"
            )
            
            cmd = [
                "ffmpeg", "-ss", str(seg_start), "-t", str(seg_dur_actual), "-i", src,
                "-vf", fc,
                "-r", str(output_fps),
                "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
                "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
                "-g", str(int(output_fps * 2)),
                "-bf", "3",
                "-an",  # No audio for segments, we'll add it back after concatenation
                "-movflags", "+faststart",
                "-y", segment_file
            ]
            
            ok, result = _run(cmd)
            if not ok:
                print(f"⚠️ Segment {seg_idx} failed: {result.stderr[:200]}")
                # Try fallback with center crop
                fallback_x = (scaled_width - crop_width) // 2
                fallback_y = (scaled_height - crop_height) // 2
                fc_fallback = (
                    f"scale={scaled_width}:{scaled_height},"
                    f"crop={crop_width}:{crop_height}:{fallback_x}:{fallback_y},"
                    "setsar=1,format=yuv420p"
                )
                cmd_fallback = [
                    "ffmpeg", "-ss", str(seg_start), "-t", str(seg_dur_actual), "-i", src,
                    "-vf", fc_fallback,
                    "-r", str(output_fps),
                    "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
                    "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
                    "-an", "-movflags", "+faststart",
                    "-y", segment_file
                ]
                ok, result = _run(cmd_fallback)
        
        # Concatenate segments
        concat_file = os.path.join(segments_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for seg_file in segment_files:
                if os.path.exists(seg_file):
                    f.write(f"file '{seg_file}'\n")
        
        concat_output = os.path.join(segments_dir, "concat_output.mp4")
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c", "copy", "-y", concat_output
        ]
        
        ok, result = _run(cmd)
        if not ok:
            print(f"❌ Concatenation failed: {result.stderr[:500]}")
            import shutil
            shutil.rmtree(segments_dir)
            return False
        
        # Copy final output
        import shutil
        shutil.copy2(concat_output, dst)
        shutil.rmtree(segments_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Smaller dynamic framing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ---- Smaller centered frame (fallback when no faces detected) -------------

def extract_smaller_centered_frame(src, dst, start, duration,
                                  source_width: int = 1920,
                                  source_height: int = 1080):
    """
    Extract smaller vertical clip with centered frame (no face tracking).
    Used as fallback when smaller frame mode is enabled but no faces are detected.
    
    Args:
        src: Source video path
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        source_width: Source video width
        source_height: Source video height
    """
    # Smaller frame dimensions
    crop_width = settings.smaller_frame_width
    crop_height = settings.smaller_frame_height
    
    try:
        # Get video FPS
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        cap.release()
        
        output_fps = settings.output_fps if settings.output_fps else int(fps)
        
        # Scale to fit smaller frame - scale larger to allow movement room
        scale_factor = crop_height / source_height
        scaled_width = int(source_width * scale_factor)
        scaled_height = int(source_height * scale_factor)
        
        # Center crop in scaled coordinates
        crop_x = (scaled_width - crop_width) // 2
        crop_y = (scaled_height - crop_height) // 2
        
        print(f"📐 Smaller centered frame: {crop_width}x{crop_height} (no faces detected, centered)")
        
        # Simple single-pass extraction with centered crop
        fc = (
            f"scale={scaled_width}:{scaled_height},"
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},"
            "setsar=1,format=yuv420p"
        )
        
        cmd = [
            "ffmpeg", "-ss", str(start), "-t", str(duration), "-i", src,
            "-vf", fc,
            "-r", str(output_fps),
            "-c:v", "libx264", "-crf", str(settings.ffmpeg_crf),
            "-preset", settings.ffmpeg_preset, "-pix_fmt", "yuv420p",
            "-g", str(int(output_fps * 2)),
            "-bf", "3",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            "-y", dst
        ]
        
        ok, result = _run(cmd)
        if not ok:
            print(f"❌ Smaller centered frame extraction failed: {result.stderr[:500]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Smaller centered frame extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ---- Dynamic facial framing with smooth tracking --------------------------

def extract_dynamic_facial_framing(src, dst, start, duration, 
                                   face_tracks: List[Dict[str, Any]],
                                   source_width: int = 1920, 
                                   source_height: int = 1080):
    """
    Extract vertical clip with dynamic facial framing that smoothly follows the face.
    
    Uses segment-based processing for true dynamic cropping that follows face movement.
    
    Args:
        src: Source video path
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        face_tracks: List of face track dictionaries with frame numbers and bboxes
                     Format: [{"frame": int, "bbox": [x, y, w, h], "confidence": float}, ...]
        source_width: Source video width
        source_height: Source video height
    """
    if not face_tracks:
        print("⚠️ No face tracks provided, falling back to center crop")
        return extract_vertical_cover(src, dst, start, duration)
    
    # Target dimensions
    crop_width = settings.vertical_width
    crop_height = settings.vertical_height
    
    try:
        # Get video FPS to calculate frame numbers
        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Default FPS
        cap.release()
        
        # Use source FPS for output if not specified (preserves smooth motion)
        output_fps = settings.output_fps if settings.output_fps else int(fps)
        
        # Calculate frame range
        start_frame = int(start * fps)
        end_frame = int((start + duration) * fps)
        total_frames = end_frame - start_frame
        
        # Sort face tracks by frame number
        face_tracks_sorted = sorted(face_tracks, key=lambda x: x["frame"])
        
        # Create a dictionary for quick lookup
        face_dict = {ft["frame"]: ft for ft in face_tracks_sorted}
        
        # Calculate scale factor to convert horizontal to vertical
        # We need to scale so we can crop a vertical frame
        # Scale factor: make height match crop_height, then crop width to crop_width
        scale_factor = crop_height / source_height
        scaled_width = int(source_width * scale_factor)
        scaled_height = int(source_height * scale_factor)  # Should equal crop_height
        
        # Smooth interpolation for crop positions (in scaled coordinates)
        crop_positions = []
        # Start with center crop in scaled coordinates
        last_crop_x = (scaled_width - crop_width) // 2
        last_crop_y = 0  # Height is already crop_height after scaling
        
        # Interpolate crop positions for all frames
        for frame_idx in range(total_frames):
            global_frame = start_frame + frame_idx
            
            # Get face position for this frame (or interpolate)
            if global_frame in face_dict:
                face_data = face_dict[global_frame]
                bbox = face_data["bbox"]
                face_center_x = bbox[0] + bbox[2] // 2
                face_center_y = bbox[1] + bbox[3] // 2
            else:
                # Find nearest face detection
                if face_dict:
                    nearest_frame = min(face_dict.keys(), key=lambda x: abs(x - global_frame))
                    frame_diff = abs(nearest_frame - global_frame)
                    max_interpolation_frames = int(fps * 1.0)  # Interpolate up to 1 second away
                    
                    if frame_diff < max_interpolation_frames:
                        face_data = face_dict[nearest_frame]
                        bbox = face_data["bbox"]
                        face_center_x = bbox[0] + bbox[2] // 2
                        face_center_y = bbox[1] + bbox[3] // 2
                    else:
                        # No face nearby, use last position (hold)
                        # Convert last crop position back to source coordinates to get face position
                        face_center_x = (last_crop_x + crop_width // 2) / scale_factor
                        face_center_y = source_height // 2
                else:
                    # No face data at all, use center
                    face_center_x = source_width // 2
                    face_center_y = source_height // 2
            
            # Convert face center to scaled coordinates
            scaled_face_center_x = face_center_x * scale_factor
            scaled_face_center_y = face_center_y * scale_factor
            
            # Calculate target crop position (center face horizontally in crop)
            # Vertical position is always 0 since scaled height equals crop height
            target_crop_x = scaled_face_center_x - crop_width // 2
            target_crop_y = 0  # Always crop from top after scaling
            
            # Apply bounds in scaled coordinates
            target_crop_x = max(0, min(target_crop_x, scaled_width - crop_width))
            target_crop_y = 0  # Fixed at 0
            
            # Smooth interpolation using exponential moving average
            smoothing = settings.crop_smoothing_factor
            crop_x = int(last_crop_x * (1 - smoothing) + target_crop_x * smoothing)
            crop_y = 0  # Always 0
            
            # Ensure we stay in bounds
            crop_x = max(0, min(crop_x, scaled_width - crop_width))
            
            crop_positions.append((crop_x, crop_y))
            last_crop_x = crop_x
            last_crop_y = crop_y
        
        # Apply additional smoothing pass to reduce jitter
        # Use a moving average filter for even smoother motion
        # Use a smaller window for more responsive but still smooth motion
        if len(crop_positions) > 1:
            smoothed_positions = []
            window_size = max(3, int(fps * 0.08))  # ~0.08 second window (slightly smaller)
            half_window = window_size // 2
            
            for i in range(len(crop_positions)):
                # Get window of positions around current frame
                start_idx = max(0, i - half_window)
                end_idx = min(len(crop_positions), i + half_window + 1)
                window = crop_positions[start_idx:end_idx]
                
                # Simple moving average (simpler = better for smooth motion)
                avg_x = sum(x for x, y in window) / len(window)
                avg_y = sum(y for x, y in window) / len(window)
                
                smoothed_positions.append((int(avg_x), int(avg_y)))
            
            crop_positions = smoothed_positions
            print(f"✨ Applied smoothing filter (window: {window_size} frames)")
        
        print(f"🎬 Generating dynamic facial framing with {len(face_tracks)} face detections")
        print(f"📊 Crop positions calculated: {len(crop_positions)} frames")
        print(f"📐 Source dimensions: {source_width}x{source_height}, Crop: {crop_width}x{crop_height}")
        print(f"🎞️  Source FPS: {fps:.2f}, Output FPS: {output_fps}")
        
        # Use very small segments for smooth interpolation
        # Smaller segments = smoother transitions but more processing
        segment_duration = settings.segment_duration  # Very small segments (0.05s = 3 frames at 60fps)
        segments_dir = tempfile.mkdtemp(prefix="facial_framing_segments_")
        segment_files = []
        
        try:
            num_segments = int(math.ceil(duration / segment_duration))
            print(f"📦 Processing {num_segments} segments ({segment_duration}s each) for smooth interpolation")
            
            for seg_idx in range(num_segments):
                seg_start = start + (seg_idx * segment_duration)
                seg_dur = min(segment_duration, duration - (seg_idx * segment_duration))
                
                if seg_dur <= 0:
                    break
                
                # Calculate frame indices for this segment
                seg_start_frame_idx = int((seg_start - start) * fps)
                seg_end_frame_idx = int((seg_start + seg_dur - start) * fps)
                
                # Clamp indices
                seg_start_frame_idx = max(0, min(seg_start_frame_idx, len(crop_positions) - 1))
                seg_end_frame_idx = max(0, min(seg_end_frame_idx, len(crop_positions) - 1))
                
                # Get crop positions at segment boundaries
                start_crop_x, start_crop_y = crop_positions[seg_start_frame_idx]
                end_crop_x, end_crop_y = crop_positions[seg_end_frame_idx]
                
                # Use start position (small segments + smoothing = smooth transitions)
                # The small segment size means transitions between segments are minimal
                crop_x, crop_y = start_crop_x, start_crop_y
                
                # Create segment file
                segment_file = os.path.join(segments_dir, f"segment_{seg_idx:04d}.mp4")
                segment_files.append(segment_file)
                
                # For vertical output from horizontal source: scale height to crop_height, then crop width
                # Calculate scale factor: scale height to match crop_height
                scale_factor = crop_height / source_height
                scaled_width = int(source_width * scale_factor)
                scaled_height = crop_height  # After scaling, height equals crop_height
                
                # Crop coordinates are already in scaled space
                scaled_crop_x = crop_x
                scaled_crop_y = 0  # Always crop from top after scaling
                
                # Ensure crop coordinates are valid
                scaled_crop_x = max(0, min(scaled_crop_x, scaled_width - crop_width))
                
                # Build filter: scale height to crop_height, crop width to crop_width at crop_x position
                # Use bilinear scaling (good quality, fast) - lanczos can be overkill and cause artifacts
                fc = (
                    f"scale={scaled_width}:{scaled_height},"
                    f"crop={crop_width}:{crop_height}:{scaled_crop_x}:0,"
                    "setsar=1,"
                    "format=yuv420p"
                )
                
                cmd = [
                    "ffmpeg", "-ss", str(seg_start), "-t", str(seg_dur), "-i", src,
                    "-vf", fc,
                    "-r", str(output_fps),  # Use calculated output FPS
                    "-c:v", "libx264",
                    "-crf", str(settings.ffmpeg_crf),
                    "-preset", settings.ffmpeg_preset,
                    "-pix_fmt", "yuv420p",
                    "-g", str(int(output_fps * 2)),  # Keyframe interval (2 seconds worth of frames)
                    "-bf", "3",  # B-frames for better compression and smoother motion
                    "-an",  # No audio for segments, we'll add it back in final concat
                    "-movflags", "+faststart",
                    "-y", segment_file
                ]
                
                ok, result = _run(cmd)
                if not ok:
                    print(f"⚠️ Segment {seg_idx} failed: {result.stderr[:500]}")
                    # Use fallback: center crop with scale
                    scale_factor = crop_height / source_height
                    scaled_width = int(source_width * scale_factor)
                    center_crop_x = (scaled_width - crop_width) // 2
                    fc = (
                        f"scale={scaled_width}:{crop_height},"
                        f"crop={crop_width}:{crop_height}:{center_crop_x}:0,"
                        "setsar=1,format=yuv420p"
                    )
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
                    if not ok:
                        print(f"❌ Segment {seg_idx} fallback also failed")
                        return False
            
            # Create concat file list
            concat_file = os.path.join(segments_dir, "concat_list.txt")
            with open(concat_file, 'w') as f:
                for seg_file in segment_files:
                    if os.path.exists(seg_file):
                        f.write(f"file '{seg_file}'\n")
            
            # Concatenate segments
            concat_output = os.path.join(segments_dir, "concat_output.mp4")
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c", "copy",
                "-y", concat_output
            ]
            
            ok, result = _run(cmd)
            if not ok:
                print(f"❌ Concatenation failed: {result.stderr[:500]}")
                return False
            
            # Add audio back and create final output with proper sync
            cmd = [
                "ffmpeg", "-i", concat_output,
                "-ss", str(start), "-t", str(duration), "-i", src,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264",
                "-crf", str(settings.ffmpeg_crf),
                "-preset", settings.ffmpeg_preset,
                "-r", str(output_fps),  # Ensure consistent frame rate
                "-g", "60",  # Keyframe interval (match to FPS)
                "-bf", "3",  # B-frames for smoother motion
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                "-shortest",
                "-vsync", "cfr",  # Constant frame rate for smooth playback
                "-movflags", "+faststart",
                "-y", dst
            ]
            
            ok, result = _run(cmd)
            if not ok:
                print(f"❌ Final audio merge failed: {result.stderr[:500]}")
                # Try without audio
                cmd = [
                    "ffmpeg", "-i", concat_output,
                    "-c:v", "copy",
                    "-movflags", "+faststart",
                    "-y", dst
                ]
                ok, result = _run(cmd)
                if not ok:
                    return False
            
            return True
            
        finally:
            # Cleanup segment files
            import shutil
            try:
                shutil.rmtree(segments_dir)
            except Exception as e:
                print(f"⚠️ Failed to cleanup segments directory: {e}")
        
    except Exception as e:
        print(f"❌ Error in dynamic facial framing: {e}")
        import traceback
        traceback.print_exc()
        return False

def _generate_dynamic_crop_filter(crop_positions: List[Tuple[int, int]], 
                                  fps: float,
                                  source_width: int,
                                  source_height: int,
                                  crop_width: int,
                                  crop_height: int) -> str:
    """
    Generate an ffmpeg filter expression for dynamic cropping.
    
    Since ffmpeg doesn't easily support per-frame crop from file, we use
    a segment-based approach or generate a complex expression.
    
    For smooth dynamic cropping, we'll use the crop filter with time-based
    interpolation between keyframes.
    """
    # Create keyframes for interpolation (every N frames)
    keyframe_interval = max(1, int(fps * 0.33))  # Keyframe every 0.33 seconds
    keyframes = []
    
    for i in range(0, len(crop_positions), keyframe_interval):
        crop_x, crop_y = crop_positions[i]
        time_sec = i / fps
        keyframes.append((time_sec, crop_x, crop_y))
    
    # Always include last frame
    if keyframes[-1][0] != (len(crop_positions) - 1) / fps:
        crop_x, crop_y = crop_positions[-1]
        time_sec = (len(crop_positions) - 1) / fps
        keyframes.append((time_sec, crop_x, crop_y))
    
    # Generate filter expression using linear interpolation
    # Format: crop=w:h:x:y where x and y are expressions based on time
    # We'll use the 'between' function to create piecewise linear interpolation
    
    if len(keyframes) == 1:
        # Single keyframe, use constant crop
        crop_x, crop_y = keyframes[0][1], keyframes[0][2]
        return f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={crop_width}:{crop_height},setsar=1,format=yuv420p"
    
    # Build interpolation expression
    # For each segment between keyframes, use linear interpolation
    expr_x_parts = []
    expr_y_parts = []
    
    for i in range(len(keyframes) - 1):
        t1, x1, y1 = keyframes[i]
        t2, x2, y2 = keyframes[i + 1]
        
        # Linear interpolation: x = x1 + (x2 - x1) * (t - t1) / (t2 - t1)
        if t2 > t1:
            slope_x = (x2 - x1) / (t2 - t1)
            slope_y = (y2 - y1) / (t2 - t1)
            
            # FFmpeg expression for this segment
            if i == 0:
                condition = f"between(t,{t1},{t2})"
            else:
                condition = f"between(t,{t1},{t2})"
            
            interp_x = f"if({condition},{x1}+({x2}-{x1})*((t-{t1})/({t2}-{t1})),"
            interp_y = f"if({condition},{y1}+({y2}-{y1})*((t-{t1})/({t2}-{t1})),"
            
            expr_x_parts.append(interp_x)
            expr_y_parts.append(interp_y)
        else:
            # Constant value
            expr_x_parts.append(f"if(between(t,{t1},{t2}),{x1},")
            expr_y_parts.append(f"if(between(t,{t1},{t2}),{y1},")
    
    # Close all nested if statements and add final value
    final_x, final_y = keyframes[-1][1], keyframes[-1][2]
    expr_x = "".join(expr_x_parts) + str(final_x) + ")" * len(expr_x_parts)
    expr_y = "".join(expr_y_parts) + str(final_y) + ")" * len(expr_y_parts)
    
    # Use crop filter with expressions
    # Note: FFmpeg's crop filter doesn't directly support expressions for x and y
    # We need to use a different approach
    
    # Alternative: Use the 'crop' filter with 'enable' and segment the video
    # Or use a Python script to generate individual crop commands
    
    # Best practical approach: Use a simpler method with fewer keyframes
    # and let ffmpeg's internal interpolation handle smoothing
    
    # For now, let's use a segment-based approach that's more reliable
    # This will be handled differently - we'll return a placeholder and
    # implement segment-based cropping in the main function
    
    # Actually, let's use a Python-based approach: generate multiple segments
    # and concatenate them. But that's complex. Instead, let's use a single
    # filter that averages nearby frames or uses a moving average.
    
    # Simplest working solution: Use the first, middle, and last crop positions
    # and let ffmpeg interpolate, or use a script filter
    
    # For practical implementation, we'll use the 'crop' filter with enable expressions
    # that switch between different crop positions at different times
    
    # Since FFmpeg expressions can get complex, let's use a script-based approach:
    # Create a filter script file that ffmpeg can read
    
    # Actually, the best approach for dynamic cropping in ffmpeg is to use
    # the 'crop' filter with the 'enable' option and time-based expressions,
    # but crop x,y can't be expressions directly in older ffmpeg versions.
    
    # Modern solution: Use ffmpeg's 'crop' with variables and the 'geq' filter
    # Or use the 'cropdetect' filter output to generate a script
    
    # For now, let's implement a practical solution using segment-based cropping
    # Return a filter that uses the median crop position for the entire clip
    # The full dynamic version will require segment-based processing
    
    # Calculate average crop position (centroid of all positions)
    avg_x = int(sum(x for x, y in crop_positions) / len(crop_positions))
    avg_y = int(sum(y for x, y in crop_positions) / len(crop_positions))
    
    # Ensure bounds
    avg_x = max(0, min(avg_x, source_width - crop_width))
    avg_y = max(0, min(avg_y, source_height - crop_height))
    
    # For true dynamic cropping, we need to process in segments
    # Return a filter that can be used, but note that full dynamic requires segments
    return f"crop={crop_width}:{crop_height}:{avg_x}:{avg_y},scale={crop_width}:{crop_height},setsar=1,format=yuv420p"

# ---- Dynamic layout rendering with state machine --------------------------

def extract_dynamic_layout(src, dst, start, duration, state_machine: LayoutStateMachine, 
                          background_mode: str = "blur", game_bg_path: Optional[str] = None,
                          use_segmented_mode: bool = True, podcast_bg_path: Optional[str] = None,
                          use_smaller_frame: bool = False) -> Dict[str, Any]:
    """
    Extract vertical clip using dynamic layout switching based on state machine.
    
    Uses segment-based mode switching for mixed content (gameplay + face):
    - Processes video in segments
    - Switches between cover crop and face tracking based on content in each segment
    - Ignores corner webcams, only tracks main subject faces
    - Detects podcasts (high face detection, no gameplay) and uses background video
    - Supports smaller frame mode with vertical tracking
    
    Args:
        src: Source video path
        dst: Destination video path
        start: Start time in seconds
        duration: Duration in seconds
        state_machine: Layout state machine instance
        background_mode: "blur", "gameplay", or "podcast"
        game_bg_path: Path to gameplay background video
        use_segmented_mode: If True, use segment-based mode switching (recommended for mixed content)
        podcast_bg_path: Path to podcast background video (default: settings.podcast_background_video_path)
        use_smaller_frame: If True, use smaller frame mode with vertical tracking
        
    Returns:
        Dict with success status and layout information
    """
    # Check if podcast background video exists
    if podcast_bg_path is None:
        podcast_bg_path = settings.podcast_background_video_path
    if not os.path.exists(podcast_bg_path):
        podcast_bg_path = None
    
    # Use segment-based mode switching for mixed content
    if use_segmented_mode and duration > settings.segment_analysis_duration:
        from app.vertical_segmented import extract_segmented_dynamic_layout
        return extract_segmented_dynamic_layout(
            src, dst, start, duration, state_machine,
            background_mode, settings.segment_analysis_duration,
            podcast_bg_path=podcast_bg_path,
            use_smaller_frame=use_smaller_frame
        )
    
    # Fallback to original single-mode analysis
    print(f"🎬 Dynamic layout extraction: {start:.2f}s to {start + duration:.2f}s")
    
    # Get video info for frame analysis
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"❌ Could not open video: {src}")
        return {"success": False, "error": "Could not open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range for this clip
    start_frame = int(start * fps)
    end_frame = int((start + duration) * fps)
    
    print(f"📊 Frame range: {start_frame} to {end_frame} (fps: {fps})")
    
    # Analyze frames to determine layout states and track faces
    layout_states = []
    face_tracks = []
    current_frame = start_frame
    
    while current_frame < end_frame and current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Get frame dimensions for main subject detection
        frame_height, frame_width = frame.shape[:2]
        
        # Update state machine with current frame
        state = state_machine.update_state(frame, current_frame, background_mode)
        
        # Get layout configuration
        layout_config = state_machine.get_current_layout_config()
        
        layout_states.append({
            "frame": current_frame,
            "state": state.value,
            "config": layout_config
        })
        
        # Collect face tracks for dynamic framing (only main subject faces)
        # Only add faces that are confirmed as main subject (not corner webcams)
        if layout_config.get("primary_face"):
            face_track = layout_config["primary_face"]
            bbox = face_track.get("bbox", [0, 0, 0, 0])
            
            # Double-check this is a main subject face (not corner webcam)
            # Verify this face meets main subject criteria
            face_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            frame_area = frame_width * frame_height
            face_area_percentage = (face_area / frame_area * 100) if frame_area > 0 else 0
            
            # Check if face is large enough and not in corner
            if face_area_percentage >= settings.min_face_area_percentage:
                # Check position (not in corner)
                face_center_x = bbox[0] + bbox[2] / 2 if len(bbox) >= 3 else 0
                face_center_y = bbox[1] + bbox[3] / 2 if len(bbox) >= 4 else 0
                
                corner_threshold = settings.corner_webcam_threshold
                is_in_corner = (
                    (face_center_x < frame_width * corner_threshold or 
                     face_center_x > frame_width * (1 - corner_threshold)) and
                    (face_center_y < frame_height * corner_threshold or 
                     face_center_y > frame_height * (1 - corner_threshold))
                )
                
                # Only add if not in corner (main subject)
                if not is_in_corner:
                    face_tracks.append({
                        "frame": current_frame,
                        "bbox": bbox,
                        "confidence": face_track.get("confidence", 0.5)
                    })
        
        current_frame += 1
    
    cap.release()
    
    # Determine dominant layout for this clip
    state_counts = {}
    for ls in layout_states:
        state = ls["state"]
        state_counts[state] = state_counts.get(state, 0) + 1
    
    dominant_state = max(state_counts.items(), key=lambda x: x[1])[0]
    total_frames_analyzed = sum(state_counts.values())
    face_detection_percentage = (len(face_tracks) / total_frames_analyzed * 100) if total_frames_analyzed > 0 else 0
    
    print(f"🎯 Dominant layout state: {dominant_state} ({state_counts[dominant_state]} frames)")
    print(f"👤 Main subject face detection rate: {face_detection_percentage:.1f}% ({len(face_tracks)}/{total_frames_analyzed} frames)")
    
    # Extract clip using dominant layout
    success = False
    
    # Check if we have sufficient MAIN SUBJECT face detections for dynamic framing
    # Corner webcams are filtered out, only main subject faces count
    has_sufficient_faces = (face_detection_percentage >= settings.min_face_detection_percentage and 
                           len(face_tracks) >= settings.min_face_detections)
    
    if dominant_state == LayoutState.COVER.value or not has_sufficient_faces:
        # No main subject human/speaker in frame - use simple vertical cover crop
        # This handles: gameplay, corner webcams, landscape videos, or any non-main-subject content
        if len(face_tracks) > 0:
            print(f"🎮 Corner webcam detected, but no main subject face ({face_detection_percentage:.1f}% main subject detection)")
            print(f"📐 Using simple vertical cover crop (ignoring corner webcam)")
        else:
            print(f"🎮 No main subject human detected ({face_detection_percentage:.1f}% face detection)")
            print(f"📐 Using simple vertical cover crop (original horizontal-to-vertical conversion)")
        success = extract_vertical_cover(src, dst, start, duration)
        
    elif dominant_state == LayoutState.VERT_FOCUS.value and face_tracks and has_sufficient_faces:
        # Use dynamic facial framing if we have sufficient MAIN SUBJECT face tracks
        print(f"👤 Main subject face detected! Using dynamic facial framing with {len(face_tracks)} face detections")
        print(f"📊 Filtered out corner webcams - only tracking main subject")
        # Get source video dimensions
        source_width, source_height = get_video_dimensions(src)
        success = extract_dynamic_facial_framing(src, dst, start, duration, face_tracks,
                                                 source_width, source_height)
        
    elif dominant_state == LayoutState.VERT_FOCUS.value and has_sufficient_faces:
        # Fallback to static face-centered crop
        print(f"👤 Using static face-centered crop")
        primary_face = None
        for ls in layout_states:
            if ls["state"] == LayoutState.VERT_FOCUS.value and ls["config"].get("primary_face"):
                primary_face = FaceTrack(
                    track_id=ls["config"]["primary_face"]["track_id"],
                    bbox=ls["config"]["primary_face"]["bbox"],
                    confidence=0.8
                )
                break
        
        success = extract_podcast_face(src, dst, start, duration, primary_face)
        
    elif dominant_state == LayoutState.GAMEPLAY.value:
        # Gameplay background mode (when background_mode="gameplay")
        print(f"🎮 Using gameplay background mode")
        success = extract_gameplay_background(src, dst, start, duration, game_bg_path)
        
    else:  # BG_BLUR_RECT (fallback, should rarely happen now)
        print(f"🖼️  Using blurred background layout")
        success = extract_blur_background(src, dst, start, duration)
    
    # Get state timeline for manifest
    state_timeline = state_machine.get_state_timeline()
    
    return {
        "success": success,
        "dominant_state": dominant_state,
        "state_counts": state_counts,
        "state_timeline": state_timeline,
        "layout_states": layout_states,
        "face_tracks": face_tracks
    }

# ---- public API ------------------------------------------------------------

def extract_vertical_clip(src, dst, start, duration, mode="cover", theme=None, bg_roots=None):
    """
    mode: "cover" | "podcast_face" | "gaming_template" | "blur_background" | "dynamic"
    theme (gaming): "subway", "templerun", "minecraft" (used to pick bg file)
    bg_roots: dict like {"subway": "assets/bg/subway.mp4", ...}
    """
    print(f"🎬 Extracting vertical clip: mode={mode}, theme={theme}")
    
    if mode == "cover":
        success = extract_vertical_cover(src, dst, start, duration)
        if not success:
            print(f"❌ Vertical cover extraction failed - NOT falling back to standard")
            return False
        return True
        
    elif mode == "podcast_face":
        success = extract_podcast_face(src, dst, start, duration)
        if not success:
            print(f"❌ Podcast face extraction failed - NOT falling back to standard")
            return False
        return True
        
    elif mode == "gaming_template":
        if not bg_roots or not theme or theme not in bg_roots:
            print(f"⚠️ Gaming theme '{theme}' not found in bg_roots, using blur background")
            return extract_blur_background(src, dst, start, duration)
        
        bg_path = bg_roots[theme]
        if not os.path.exists(bg_path):
            print(f"⚠️ Game background file not found: {bg_path}, using blur background")
            return extract_blur_background(src, dst, start, duration)
        
        success = extract_gameplay_background(src, dst, start, duration, bg_path)
        if not success:
            print(f"❌ Gameplay background extraction failed - NOT falling back to standard")
            return False
        return True
        
    elif mode == "blur_background":
        return extract_blur_background(src, dst, start, duration)
        
    elif mode == "dynamic":
        # Dynamic layout requires state machine - this should be called from video processor
        print(f"⚠️ Dynamic mode requires state machine - use extract_dynamic_layout instead")
        return extract_vertical_cover(src, dst, start, duration)
        
    else:
        print(f"⚠️ Unknown mode '{mode}', using cover")
        return extract_vertical_cover(src, dst, start, duration)

# ---- legacy compatibility functions ----------------------------------------

def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """
    Get video dimensions using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 1920, 1080  # Default fallback
        
        data = json.loads(result.stdout)
        if not data.get('streams'):
            return 1920, 1080
        
        stream = data['streams'][0]
        width = int(stream.get('width', 1920))
        height = int(stream.get('height', 1080))
        
        return width, height
        
    except Exception as e:
        print(f"⚠️ Failed to get video dimensions: {e}")
        return 1920, 1080  # Default fallback

def get_vertical_dimensions(width: int = None) -> Tuple[int, int]:
    """
    Get vertical video dimensions based on settings.
    
    Args:
        width: Target width (defaults to settings)
        
    Returns:
        Tuple of (width, height)
    """
    if width is None:
        width = settings.vertical_width
    
    # Use configured height or calculate based on 9:16 aspect ratio
    if hasattr(settings, 'vertical_height'):
        height = settings.vertical_height
    else:
        height = int(width * 16 / 9)
    
    return width, height

