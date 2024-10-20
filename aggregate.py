import os
from collections import Counter
from utils import index_to_class
def frames_to_time(frame_index, fps):
    return round(frame_index / fps, 2)

def aggregate_predictions_with_samples(predictions, fps, interval=10):
    def frames_to_time(frame_index, fps):
        return frame_index / fps

    aggregated = []
    current_label = None
    start_frame = None

    sorted_frames = sorted(predictions.keys())
    
    for frame_index in sorted_frames:
        label = predictions.get(frame_index, 'Unknown')

        # If we're starting a new label sequence
        if current_label is None:
            current_label = label
            start_frame = frame_index
        elif label != current_label:
            # If the label changes, save the previous range
            start_time = frames_to_time(start_frame, fps)
            end_time = frames_to_time(frame_index - 1, fps)  # The last frame of the previous label
            aggregated.append(f"{start_time:.2f}-{end_time:.2f}s: {current_label}")

            # Start the new label sequence
            current_label = label
            start_frame = frame_index

    # Capture the last label sequence
    if current_label is not None:
        start_time = frames_to_time(start_frame, fps)
        end_time = frames_to_time(sorted_frames[-1], fps)  # The last frame in the video
        aggregated.append(f"{start_time:.2f}-{end_time:.2f}s: {current_label}")

    return aggregated
