"""
Test script for dynamic facial framing.

Place your test video as 'input.mp4' in the project root directory,
or specify the path using the --input argument.
"""

import os
import sys
import argparse
# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from app.vertical import extract_dynamic_layout, get_video_dimensions
from app.layout.state_machine import LayoutStateMachine
from app.settings import settings


def test_dynamic_facial_framing(input_video: str, output_video: str = "output_dynamic.mp4",
                                start: float = 0.0, duration: float = None,
                                background_mode: str = "blur", use_smaller_frame: bool = False):
    """
    Test dynamic facial framing on a video.
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        start: Start time in seconds (default: 0.0)
        duration: Duration in seconds (None = entire video)
        background_mode: "blur", "gameplay", or "podcast"
        use_smaller_frame: If True, use smaller frame mode with vertical tracking
    """
    print("🎬 Dynamic Facial Framing Test")
    print("=" * 60)
    print()
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        print("💡 Please provide a valid video file path")
        return False
    
    # Get video info
    print(f"📹 Input video: {input_video}")
    width, height = get_video_dimensions(input_video)
    print(f"📐 Video dimensions: {width}x{height}")
    print()
    
    # Get video duration if not specified
    if duration is None:
        import cv2
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = frame_count / fps if fps > 0 else 0
        cap.release()
        duration = video_duration - start
        print(f"⏱️  Video duration: {video_duration:.2f} seconds")
        print(f"⏱️  Processing: {start:.2f}s to {start + duration:.2f}s")
    else:
        print(f"⏱️  Processing: {start:.2f}s to {start + duration:.2f}s")
    print()
    
    # Display current settings
    print("⚙️  Current Settings:")
    print(f"   Face detection model: {settings.face_detection_model}")
    print(f"   Crop smoothing factor: {settings.crop_smoothing_factor}")
    print(f"   Face tracking smoothing: {settings.face_tracking_smoothing}")
    if use_smaller_frame:
        print(f"   Output dimensions: {settings.smaller_frame_width}x{settings.smaller_frame_height} (smaller frame mode)")
    else:
        print(f"   Output dimensions: {settings.vertical_width}x{settings.vertical_height}")
    print(f"   Output FPS: {settings.output_fps}")
    print(f"   Background mode: {background_mode}")
    if use_smaller_frame:
        print(f"   Smaller frame mode: enabled (vertical + horizontal tracking)")
    print()
    
    # Initialize state machine
    print("🔍 Initializing face detector and state machine...")
    state_machine = LayoutStateMachine()
    print("✅ Ready to process")
    print()
    
    # Process video
    print("🎬 Processing video with dynamic facial framing...")
    print("   This may take a few minutes depending on video length...")
    print()
    
    try:
        result = extract_dynamic_layout(
            src=input_video,
            dst=output_video,
            start=start,
            duration=duration,
            state_machine=state_machine,
            background_mode=background_mode,
            use_smaller_frame=use_smaller_frame
        )
        
        print()
        print("=" * 60)
        
        if result["success"]:
            print("✅ SUCCESS! Video processed successfully!")
            print()
            print(f"📁 Output video: {output_video}")
            
            # Handle segmented mode results (different structure)
            if "num_segments" in result:
                print()
                print(f"📊 Segmented mode results:")
                print(f"   Total segments: {result['num_segments']}")
                print(f"   Cover crop segments: {result.get('cover_segments', 0)}")
                print(f"   Face tracking segments: {result.get('face_tracking_segments', 0)}")
                print(f"   Podcast segments: {result.get('podcast_segments', 0)}")
                print(f"   Smaller frame segments: {result.get('smaller_frame_segments', 0)}")
                print()
                print("💡 The video switches between modes based on content:")
                print("   - Gameplay/corner webcam segments → Cover crop")
                print("   - Main subject face segments → Dynamic facial framing")
                print("   - Podcast segments (high face detection) → Podcast background with fast teleport")
                if use_smaller_frame:
                    print("   - Smaller frame segments → Vertical + horizontal tracking")
                print()
                return True
            else:
                # Single mode results
                if "dominant_state" in result:
                    print(f"🎯 Dominant layout: {result['dominant_state']}")
                if "face_tracks" in result:
                    print(f"👤 Face tracks detected: {len(result.get('face_tracks', []))}")
                print()
            
            # Show state distribution
            if result.get('state_counts'):
                print("📊 Layout state distribution:")
                for state, count in result['state_counts'].items():
                    percentage = (count / sum(result['state_counts'].values())) * 100
                    print(f"   {state}: {count} frames ({percentage:.1f}%)")
            print()
            
            # Check if face was detected
            if result.get('face_tracks'):
                face_tracks = result['face_tracks']
                frames_with_face = len(face_tracks)
                total_frames = sum(result.get('state_counts', {}).values())
                if total_frames > 0:
                    face_percentage = (frames_with_face / total_frames) * 100
                    print(f"👤 Face detection rate: {face_percentage:.1f}%")
                    print(f"   Frames with face: {frames_with_face} / {total_frames}")
                print()
                
                # Show face track statistics
                if face_tracks:
                    confidences = [ft.get('confidence', 0) for ft in face_tracks]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    print(f"📈 Average face confidence: {avg_confidence:.2f}")
                    print()
            else:
                print("⚠️  No face tracks detected - video may use fallback layout")
                print()
            
            print("💡 Tips:")
            print("   - Check the output video to verify face tracking")
            print("   - Adjust crop_smoothing_factor in settings.py for smoother/faster tracking")
            print("   - Adjust face_detection_confidence if faces aren't detected")
            print()
            
            return True
        else:
            print("❌ FAILED! Video processing failed")
            if result.get('error'):
                print(f"   Error: {result['error']}")
            return False
            
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test dynamic facial framing on a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with input.mp4 in current directory
  python test_facial_framing.py

  # Test with custom input and output
  python test_facial_framing.py --input my_video.mp4 --output result.mp4

  # Test a specific segment (first 30 seconds)
  python test_facial_framing.py --input my_video.mp4 --start 0 --duration 30

  # Test with gameplay background mode
  python test_facial_framing.py --input my_video.mp4 --mode gameplay
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="input.mp4",
        help="Input video file path (default: input.mp4)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_dynamic.mp4",
        help="Output video file path (default: output_dynamic.mp4)"
    )
    
    parser.add_argument(
        "--start", "-s",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Duration in seconds (default: entire video)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["blur", "gameplay", "podcast"],
        default="blur",
        help="Background mode: blur, gameplay, or podcast (default: blur)"
    )
    
    parser.add_argument(
        "--smaller-frame", "-sf",
        action="store_true",
        help="Use smaller frame mode with vertical + horizontal tracking"
    )
    
    args = parser.parse_args()
    
    # Enable smaller frame mode in settings if requested
    if args.smaller_frame:
        settings.smaller_frame_enabled = True
        print("📐 Smaller frame mode enabled: vertical + horizontal tracking")
        print()
    
    # Run test
    success = test_dynamic_facial_framing(
        input_video=args.input,
        output_video=args.output,
        start=args.start,
        duration=args.duration,
        background_mode=args.mode,
        use_smaller_frame=args.smaller_frame
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

