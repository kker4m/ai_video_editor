import glob
import json
import os
import tempfile
import webrtcvad
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from moviepy.editor import VideoFileClip, concatenate_videoclips
from openai import OpenAI
from pydub import AudioSegment
from credentials import OPENAI_API_KEY
# Instantiate the client (using the new client-based API)
client = OpenAI(api_key=OPENAI_API_KEY)


def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()


def detect_segments(
    audio,
    frame_duration_ms=30,
    padding_duration_ms=300,
    aggressiveness=3,
    post_speech_padding_sec=0.2,
    **kwargs,
):
    """
    Detect speech segments using voice activity detection (VAD) via webrtcvad,
    with an adjustable post-speech padding to determine the exact cut.

    This function converts the given pydub AudioSegment to mono 16-bit PCM at a supported
    sample rate (8000, 16000, 32000, or 48000 Hz), splits it into frames, and uses webrtcvad
    to determine which frames contain speech. Contiguous speech frames are merged into segments.
    The end of each segment is adjusted by adding a post-speech padding (in seconds). Setting
    post_speech_padding_sec to 0 cuts immediately when speech stops; setting it to 1 waits 1 second
    of silence before cutting.

    Parameters:
      audio (AudioSegment): The audio to analyze.
      frame_duration_ms (int): Duration of each frame in milliseconds (must be 10, 20, or 30).
          Defaults to 30. If an alternate value is passed via the keyword 'chunk_ms', that value is used.
      padding_duration_ms (int): If the gap between speech segments is less than this (in ms),
          the segments are merged. Defaults to 300 ms.
      aggressiveness (int): VAD aggressiveness mode (0 = least, 3 = most aggressive). Defaults to 3.
      post_speech_padding_sec (float): Additional time (in seconds) to include after the last detected speech
          frame. Use 0 for an immediate cut, 1 to wait 1 second in silence.
      **kwargs: Accepts an optional 'chunk_ms' keyword to override frame_duration_ms for backward compatibility.

    Returns:
      List[Dict]: A list of dictionaries, each with 'start' and 'end' (in seconds) marking a detected speech segment.
    """
    

    # Allow backward compatibility with 'chunk_ms'
    if "chunk_ms" in kwargs:
        frame_duration_ms = kwargs["chunk_ms"]

    # webrtcvad only supports frame durations of 10, 20, or 30 ms.
    if frame_duration_ms not in (10, 20, 30):
        print(
            f"Warning: frame_duration_ms {frame_duration_ms} is invalid. Using 30 ms instead."
        )
        frame_duration_ms = 30

    # Ensure audio is mono and at a supported sample rate.
    audio = audio.set_channels(1)
    if audio.frame_rate not in (8000, 16000, 32000, 48000):
        audio = audio.set_frame_rate(16000)
    sample_rate = audio.frame_rate
    sample_width = audio.sample_width  # bytes per sample (typically 2 for 16-bit PCM)
    raw_audio = audio.raw_data

    vad = webrtcvad.Vad(aggressiveness)
    # Calculate frame size in samples and then in bytes.
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    frame_bytes = frame_size * sample_width

    # Split raw audio into frames of exact length.
    frames = []
    for i in range(0, len(raw_audio) - frame_bytes + 1, frame_bytes):
        frame = raw_audio[i : i + frame_bytes]
        timestamp = i / (sample_rate * sample_width)
        frames.append((timestamp, frame))

    # Label each frame using VAD.
    speech_flags = []
    for timestamp, frame in frames:
        try:
            is_speech = vad.is_speech(frame, sample_rate)
        except Exception as e:
            print(f"Error processing frame at {timestamp:.2f} sec: {e}")
            is_speech = False
        speech_flags.append((timestamp, is_speech))

    # Aggregate contiguous speech frames into segments,
    # and adjust the segment end by adding post_speech_padding_sec.
    segments = []
    segment_start = None
    last_speech_timestamp = None
    for timestamp, is_speech in speech_flags:
        if is_speech:
            if segment_start is None:
                segment_start = timestamp
            last_speech_timestamp = timestamp
        else:
            if segment_start is not None and last_speech_timestamp is not None:
                # End the segment at the last speech timestamp plus the post-speech padding.
                segments.append(
                    {
                        "start": segment_start,
                        "end": last_speech_timestamp + post_speech_padding_sec,
                    }
                )
                segment_start = None
                last_speech_timestamp = None
    if segment_start is not None:
        total_duration = len(raw_audio) / (sample_rate * sample_width)
        segments.append({"start": segment_start, "end": total_duration})

    # Merge segments that are separated by less than padding_duration_ms.
    merged_segments = []
    if segments:
        current = segments[0]
        for seg in segments[1:]:
            if seg["start"] - current["end"] < padding_duration_ms / 1000.0:
                current["end"] = seg["end"]
            else:
                merged_segments.append(current)
                current = seg
        merged_segments.append(current)

    return merged_segments


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def transcribe_audio_segment(segment_audio_path):
    """
    Transcribe an audio segment using Whisper via the new API.
    The response is now a pydantic model; we convert it with model_dump().
    """
    with open(segment_audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="json"
        )
    transcript_data = transcript.model_dump()
    return transcript_data.get("text", "")


def transcribe_segments(audio, segments):
    """
    For each detected segment, export its audio to a temporary file and transcribe it.
    Returns a list of dicts with keys: start, end, text.
    """
    transcriptions = []
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            segment_audio.export(tmp.name, format="wav")
            tmp_path = tmp.name
        text = transcribe_audio_segment(tmp_path)
        os.remove(tmp_path)
        transcriptions.append(
            {"start": seg["start"], "end": seg["end"], "text": text.strip()}
        )
    return transcriptions


def get_llm_suggestion(raw_transcription):
    """
    Uses LangChain with gpt-4o-mini to filter out redundant or duplicate transcription segments.
    Given a raw JSON transcription (a list of objects, each with 'start', 'end', and 'text'),
    the LLM returns a JSON object with a key "filtered_transcription" containing the filtered list.
    For segments with duplicate or nearly identical text, only keep the one with the highest start time (i.e. the last occurrence)
    and remove all earlier duplicates.
    """

    # Define the expected output schema.
    response_schemas = [
        ResponseSchema(
            name="filtered_transcription",
            description=(
                "A list of transcription segments to keep, in chronological order. "
                "Each segment is an object with 'start' (number, seconds), 'end' (number, seconds), and 'text' (string)."
            ),
        )
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    # Build a detailed prompt with explicit instructions.
    prompt = (
        "You are given a raw JSON transcription of a video as an array of objects. "
        "Each object has three keys: 'start' (a number indicating the start time in seconds), "
        "'end' (a number indicating the end time in seconds), and 'text' (a string with the transcribed speech).\n\n"
        "Your task is to remove any segments that are redundant, duplicate, or mistaken. "
        "Specifically, if two or more segments have the same or nearly identical 'text' (ignoring minor differences such as punctuation or trailing ellipses), "
        "only keep the segment with the highest start time (i.e. the last occurrence) and remove all earlier duplicates. "
        "The final output should be a JSON object with a single key 'filtered_transcription' that contains an array of the remaining segments, "
        "Observe that sometimes the segments may be rephrased, so consider this a duplication and always consider the last occurrence."
        'example of input: { "start": 6.84, "end": 9.8, "text": "In my previous video, I\'ve reached..." }, { "start": 12.24, "end": 15.08, "text": "In my previous video, I\'ve reached many comments." }, { "start": 15.84, "end": 24.17, "text": "In my previous video I\'ve received many comments asking why use an LLM to scrape if we can just use normal selenium, beautiful soup, or puppeteer." }, in this example you would only retain the last object.'
        "in chronological order.\n\n"
        "Follow exactly the format instructions provided below:\n\n"
        f"{format_instructions}\n\n"
        "Here is the raw transcription JSON:\n"
        f"{json.dumps(raw_transcription, indent=2)}"
    )

    # Use gpt-4o-mini via LangChain.
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    response = llm.predict(prompt)

    try:
        parsed_output = output_parser.parse(response)
        return parsed_output.get("filtered_transcription", raw_transcription)
    except Exception as e:
        print("Error parsing LLM output:", e)
        return raw_transcription


def create_final_video(video_path, segments):
    video = VideoFileClip(video_path)
    clips = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        if end - start > 0.1:
            clips.append(video.subclip(start, end))
    if clips:
        final_clip = concatenate_videoclips(clips)
        os.makedirs("edited", exist_ok=True)
        output_path = os.path.join("edited", os.path.basename(video_path))
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        final_clip.close()
    video.close()


def process_video(video_path):
    print(f"Processing {video_path}")
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Step 1: Extract full audio from the video
    temp_audio_file = f"{base_name}_temp_audio.wav"
    extract_audio(video_path, temp_audio_file)

    # Step 2: Load audio and detect segments based on sound levels
    audio = AudioSegment.from_file(temp_audio_file)
    raw_segments = detect_segments(audio, chunk_ms=100)
    raw_segments_file = f"{base_name}_raw_segments.json"
    save_json(raw_segments, raw_segments_file)
    print(f"Saved raw segments JSON to {raw_segments_file}")

    # Step 3: For each segment, transcribe the audio using Whisper
    raw_transcription = transcribe_segments(audio, raw_segments)
    raw_transcription_file = f"{base_name}_transcription.json"
    save_json(raw_transcription, raw_transcription_file)
    print(f"Saved raw transcription JSON to {raw_transcription_file}")

    os.remove(temp_audio_file)

    # Step 4: Send raw transcription to an LLM for filtering and save suggestion JSON locally
    suggestion = get_llm_suggestion(raw_transcription)
    suggestion_file = f"{base_name}_suggestion.json"
    save_json(suggestion, suggestion_file)
    print(f"Saved LLM suggestion JSON to {suggestion_file}")

    # Step 5: Create the final video using segments from the suggestion JSON
    create_final_video(video_path, suggestion)


def main():
    video_files = glob.glob("raw/*")
    for video_file in video_files:
        if os.path.isfile(video_file) and video_file.lower().endswith(
            (".mp4", ".mov", ".avi", ".mkv")
        ):
            process_video(video_file)


if __name__ == "__main__":
    main()
