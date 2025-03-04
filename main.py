import os
import subprocess
import whisper
from openai import OpenAI
import moviepy.editor as mp
import re
import json
from credentials import OPENAI_API_KEY

# OpenAI API anahtarını ayarlayın
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def extract_audio(video_path, output_audio):
    """Videodan sesi çıkarır."""
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y"
    subprocess.call(command, shell=True)

def transcribe_audio(audio_path):
    """Whisper kullanarak sesi metne çevirir."""
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    return result["segments"]

def detect_silence(video_path, silence_threshold=-30, min_silence_duration=0.5):
    """ffmpeg ile sessiz bölümleri tespit eder ve bunları liste olarak döndürür."""
    command = f"ffmpeg -i {video_path} -af silencedetect=noise={silence_threshold}dB:d={min_silence_duration} -f null - 2>&1"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    silence_data = []
    matches = re.findall(r'silence_start: (\d+\.\d+)|silence_end: (\d+\.\d+)', result.stdout)
    
    start = None
    for match in matches:
        silence_start, silence_end = match
        if silence_start:
            start = float(silence_start)
        elif silence_end and start is not None:
            silence_data.append({"start": start, "end": float(silence_end)})
            start = None
    
    return silence_data

def remove_repeated_sentences(sentences):
    """OpenAI API ile tekrarlayan cümleleri temizler."""
    text = "\n".join([sentence["text"] for sentence in sentences])  # 'sentence' yerine 'text' kullanıyoruz
    print(text)
    input()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """
Sen yapay zeka bir video editorusun, ve sana verilen json formatinda ki yaziya donusturulmus videonun tekrar eden kısımlarını silmekle görevlisin. Cumleler tamamen ayni olmak zorunda degil, ayni baglamda ki, anlamsiz cumleleri de silip sadece sonuncu cumleyi birakman gerekiyor.
"""}, 
            {"role": "user", "content": text}
        ]
    )
    
    cleaned_text = response.choices[0].message.content
    
    # Cümleleri zaman damgalarıyla eşleştir
    cleaned_sentences = []
    cleaned_lines = cleaned_text.split("\n")
    for i, line in enumerate(cleaned_lines):
        if line.strip():
            cleaned_sentences.append({
                "start": sentences[i]["start"],
                "end": sentences[i]["end"],
                "sentence": line.strip()
            })
    
    return cleaned_sentences


def apply_jump_cut(video_path, silence_data, output_path, buffer=0.3):
    """MoviePy ile sessiz bölümleri atlayarak jump cut uygular."""
    video = mp.VideoFileClip(video_path)
    final_clips = []
    
    current_pos = 0.0
    for silence in silence_data:
        start = silence["start"] - buffer if silence["start"] - buffer > 0 else 0
        end = silence["end"] + buffer if silence["end"] + buffer < video.duration else video.duration
        
        if current_pos < start:
            final_clips.append(video.subclip(current_pos, start))
        current_pos = end
    
    if current_pos < video.duration:
        final_clips.append(video.subclip(current_pos, video.duration))
    
    final = mp.concatenate_videoclips(final_clips)
    final.write_videofile(output_path, codec='libx264', fps=video.fps)

def main(video_path):
    audio_path = "temp_audio.wav"
    output_video = "output.mp4"
    
    # Videodan ses çıkar
    extract_audio(video_path, audio_path)
    
    # Ses dosyasını transkripte et
    sentences = transcribe_audio(audio_path)
    
    # Sesli metni tekrar eden cümleleri temizle
    cleaned_sentences = remove_repeated_sentences(sentences)
    
    # Sessiz bölümleri tespit et
    silence_data = detect_silence(video_path)
    
    # Jump cut işlemi uygula
    apply_jump_cut(video_path, silence_data, output_video)
    
    # Geçici ses dosyasını sil
    os.remove(audio_path)
    
    # Sonuçları yazdır
    print(f"İşlem tamamlandı. Çıktı video: {output_video}")
    print(f"Temizlenmiş metin:\n{json.dumps(cleaned_sentences, indent=4)}")

if __name__ == "__main__":
    main(r"D:\GitHub\ai_video_editor\test_video.mp4")
