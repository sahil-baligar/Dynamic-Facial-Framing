"""
Example script for using Dynamic Facial Framing.

This script demonstrates how to use the dynamic facial framing system
to convert horizontal videos to vertical format with face tracking.
"""

import os
import sys
from app.vertical import extract_dynamic_layout, extract_vertical_clip, get_video_dimensions
from app.layout.state_machine import LayoutStateMachine


def example_simple_cover():
    """Example: Simple vertical cover crop (no face detection)."""
    print("📹 Example 1: Simple vertical cover crop")
    print("=" * 50)
    
    input_video = "input.mp4"
    output_video = "output_cover.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return False
    
    success = extract_vertical_clip(
        src=input_video,
        dst=output_video,
        start=0.0,
        duration=30.0,
        mode="cover"
    )
    
    if success:
        print(f"✅ Output saved to: {output_video}")
    else:
        print("❌ Processing failed")
    
    return success


def example_face_centered():
    """Example: Face-centered crop (static face detection)."""
    print("\n📹 Example 2: Face-centered crop")
    print("=" * 50)
    
    input_video = "input.mp4"
    output_video = "output_face.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return False
    
    success = extract_vertical_clip(
        src=input_video,
        dst=output_video,
        start=0.0,
        duration=30.0,
        mode="podcast_face"
    )
    
    if success:
        print(f"✅ Output saved to: {output_video}")
    else:
        print("❌ Processing failed")
    
    return success


def example_dynamic_framing():
    """Example: Dynamic facial framing with state machine."""
    print("\n📹 Example 3: Dynamic facial framing")
    print("=" * 50)
    
    input_video = "input.mp4"
    output_video = "output_dynamic.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return False
    
    # Initialize state machine
    state_machine = LayoutStateMachine()
    
    # Extract with dynamic layout
    result = extract_dynamic_layout(
        src=input_video,
        dst=output_video,
        start=0.0,
        duration=30.0,
        state_machine=state_machine,
        background_mode="blur"  # or "gameplay"
    )
    
    if result["success"]:
        print(f"✅ Output saved to: {output_video}")
        print(f"📊 Dominant layout: {result['dominant_state']}")
        print(f"👤 Face tracks: {len(result.get('face_tracks', []))}")
        print(f"📈 State counts: {result.get('state_counts', {})}")
    else:
        print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    return result["success"]


def example_blur_background():
    """Example: Blurred background with foreground rectangle."""
    print("\n📹 Example 4: Blurred background")
    print("=" * 50)
    
    input_video = "input.mp4"
    output_video = "output_blur.mp4"
    
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return False
    
    success = extract_vertical_clip(
        src=input_video,
        dst=output_video,
        start=0.0,
        duration=30.0,
        mode="blur_background"
    )
    
    if success:
        print(f"✅ Output saved to: {output_video}")
    else:
        print("❌ Processing failed")
    
    return success


def main():
    """Run all examples."""
    print("🎬 Dynamic Facial Framing - Examples")
    print("=" * 50)
    print()
    
    # Check if input video exists
    input_video = "input.mp4"
    if not os.path.exists(input_video):
        print(f"⚠️  Input video not found: {input_video}")
        print("📝 Please provide an input video file named 'input.mp4'")
        print()
        print("Usage:")
        print("  python example.py")
        print()
        print("Or run individual examples:")
        print("  python example.py --simple")
        print("  python example.py --face")
        print("  python example.py --dynamic")
        print("  python example.py --blur")
        return
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "--simple" or mode == "-s":
            example_simple_cover()
        elif mode == "--face" or mode == "-f":
            example_face_centered()
        elif mode == "--dynamic" or mode == "-d":
            example_dynamic_framing()
        elif mode == "--blur" or mode == "-b":
            example_blur_background()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: --simple, --face, --dynamic, --blur")
    else:
        # Run all examples
        print("Running all examples...")
        print()
        
        example_simple_cover()
        example_face_centered()
        example_dynamic_framing()
        example_blur_background()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")


if __name__ == "__main__":
    main()

