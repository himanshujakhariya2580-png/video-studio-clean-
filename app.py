import gradio as gr
import cv2
import numpy as np
import subprocess
import os
import textwrap
from rembg import remove, new_session
from PIL import Image
from faster_whisper import WhisperModel

# Load models
bg_session = new_session("u2netp")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def escape_ffmpeg_text(text):
    return text.replace(":", "\\:").replace("'", "").replace('"', "").replace("%", "\\%")

def generate_viral_caption(transcript_text):
    if not transcript_text or len(transcript_text.strip()) < 5:
        return (
            "🔥 Watch till the end — you won't believe what happens next! 👀\n\n"
            "Save this video for later & share with someone who needs to see this! 💯\n\n"
            "#viral #trending #shorts #reels #fyp #explore #mustwatch #mindset"
        )
    clean_speech = transcript_text.strip()
    first_line = clean_speech.split(".")[0][:90].strip()
    words = [w.strip(".,!?").lower() for w in clean_speech.split() if len(w) > 4]
    unique_tags = list(dict.fromkeys(words))[:6]
    dynamic_hashtags = " ".join([f"#{tag}" for tag in unique_tags])
    return (
        f"🚨 {first_line}... (Wait till the end! 👀)\n\n"
        f"💡 Key Takeaway:\n\"{clean_speech[:220]}...\"\n\n"
        f"👇 Comment your thoughts below & share this with a friend!\n\n"
        f"{dynamic_hashtags} #viral #trending #reels #shorts #explorepage #fyp"
    )

def process_video(
    input_video,
    enable_bg_change,
    bg_color_choice,
    enable_shorts_mode,
    shorts_max_duration,
    enable_remove_old_hook,
    old_hook_area,
    old_hook_height_pct,
    enable_new_hook,
    hook_mode,
    custom_hook_text,
    hook_text_color,
    enable_viral_caption
):
    if not input_video:
        return None, "", "❌ Pehle apna video upload karein!"

    log = ["🚀 Processing shuru ho rahi hai..."]
    current_video = input_video
    bg_rgb = (0, 0, 0) if bg_color_choice == "Black" else (255, 255, 255)
    hex_bg = "black" if bg_color_choice == "Black" else "white"

    try:
        # 1. Long-to-Shorts Trim
        if enable_shorts_mode:
            trimmed_video = "step1_trimmed.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", current_video,
                "-t", str(int(shorts_max_duration)),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", trimmed_video
            ], check=True)
            current_video = trimmed_video
            log.append(f"✔ [ON] Video {int(shorts_max_duration)}s Shorts limit mein trim hua.")

        # 2. Remove Old Hook
        if enable_remove_old_hook:
            cleaned_video = "step2_no_old_hook.mp4"
            frac = old_hook_height_pct / 100.0
            if old_hook_area == "Top Area":
                vf_crop = f"crop=iw:ih*(1-{frac}):0:ih*{frac}"
            elif old_hook_area == "Bottom Area":
                vf_crop = f"crop=iw:ih*(1-{frac}):0:0"
            else:
                vf_crop = f"crop=iw:ih*(1-{frac*2}):0:ih*{frac}"

            subprocess.run([
                "ffmpeg", "-y", "-i", current_video,
                "-vf", vf_crop,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "copy", cleaned_video
            ], check=True)
            current_video = cleaned_video
            log.append(f"✔ [ON] Purana hook ({old_hook_area}) hataya gaya.")

        # 3. AI Background Change (Black / White)
        if enable_bg_change:
            log.append(f"🎨 AI Background remove karke '{bg_color_choice}' lagaya ja raha hai...")
            cap = cv2.VideoCapture(current_video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            temp_no_audio = "step3_bg_no_audio.mp4"
            out = cv2.VideoWriter(temp_no_audio, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            bg_layer = np.full((height, width, 3), bg_rgb, dtype=np.uint8)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                fg_rgba = remove(Image.fromarray(rgb_frame), session=bg_session)
                fg_np = np.array(fg_rgba)
                alpha = fg_np[:, :, 3:4] / 255.0
                blended = (fg_np[:, :, :3] * alpha + bg_layer * (1 - alpha)).astype(np.uint8)
                out.write(cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()

            bg_with_audio = "step3_bg_done.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_no_audio, "-i", current_video,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", bg_with_audio
            ], check=True)
            current_video = bg_with_audio
            log.append(f"✔ [ON] Background '{bg_color_choice}' set ho gaya.")

        # 4. 9:16 Vertical Shorts Canvas
        if enable_shorts_mode:
            shorts_canvas = "step4_shorts_canvas.mp4"
            vf_shorts = (
                f"scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color={hex_bg}"
            )
            subprocess.run([
                "ffmpeg", "-y", "-i", current_video,
                "-vf", vf_shorts,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "copy", shorts_canvas
            ], check=True)
            current_video = shorts_canvas
            log.append("✔ [ON] 9:16 Vertical Shorts format ready.")

        # 5. Speech Analysis
        transcript_text = ""
        if enable_viral_caption or (enable_new_hook and hook_mode == "Auto-Generate from Video"):
            log.append("🎙 Audio analyze ho rahi hai...")
            segments, _ = whisper_model.transcribe(current_video)
            transcript_text = " ".join([s.text.strip() for s in list(segments)])

        # 6. New Bold Hook (White or Black)
        if enable_new_hook:
            final_hook_video = "final_output.mp4"
            if hook_mode == "Custom Hook Text" and custom_hook_text.strip():
                raw_hook = custom_hook_text.strip().upper()
            elif transcript_text:
                raw_hook = transcript_text[:55].upper() + "..."
            else:
                raw_hook = "WATCH THIS TILL THE END!"

            wrapped_hook = "\n".join(textwrap.wrap(raw_hook, width=20))
            safe_hook = escape_ffmpeg_text(wrapped_hook)

            if hook_text_color == "White":
                font_col, border_col, box_col = "white", "black", "black@0.55"
            else:
                font_col, border_col, box_col = "black", "white", "white@0.75"

            bold_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font_opt = f"fontfile={bold_font}:" if os.path.exists(bold_font) else ""

            drawtext_filter = (
                f"drawtext={font_opt}text='{safe_hook}':"
                f"fontcolor={font_col}:fontsize=52:borderw=3:bordercolor={border_col}:"
                f"box=1:boxcolor={box_col}:boxborderw=18:"
                f"x=(w-text_w)/2:y=h*0.10"
            )
            subprocess.run([
                "ffmpeg", "-y", "-i", current_video,
                "-vf", drawtext_filter,
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "copy", final_hook_video
            ], check=True)
            current_video = final_hook_video
            log.append(f"✔ [ON] Naya Bold Hook ({hook_text_color}) lagaya gaya.")

        # 7. Viral Caption
        viral_caption_out = generate_viral_caption(transcript_text) if enable_viral_caption else "Caption OFF hai."
        if enable_viral_caption:
            log.append("✔ [ON] Viral Caption & Hashtags ready!")

        log.append("\n🎉 Final Video Ready!")
        return current_video, viral_caption_out, "\n".join(log)

    except Exception as e:
        return None, "", f"❌ Error:\n{str(e)}"

with gr.Blocks(title="AI Video Studio Pro") as demo:
    gr.Markdown("# 🎬 AI Video Studio Pro")
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="1. Apna Video Upload Karein")
            with gr.Accordion("🎨 1. AI Background Changer", open=True):
                chk_bg = gr.Checkbox(label="Background Change (ON / OFF)", value=True)
                bg_choice = gr.Radio(["Black", "White"], value="Black", label="Background Color")
            with gr.Accordion("📱 2. Long Video to Shorts", open=True):
                chk_shorts = gr.Checkbox(label="Convert to 9:16 Shorts (ON / OFF)", value=True)
                sld_duration = gr.Slider(15, 60, value=60, step=5, label="Max Duration (Seconds)")
            with gr.Accordion("🧹 3. Old Hook Remove + New Bold Hook", open=True):
                chk_rem_hook = gr.Checkbox(label="Purana Hook Remove Karein (ON / OFF)", value=False)
                old_hook_pos = gr.Radio(["Top Area", "Bottom Area", "Top & Bottom"], value="Top Area", label="Purana Hook Position")
                sld_crop_pct = gr.Slider(5, 25, value=14, step=1, label="Crop Area (%)")
                chk_new_hook = gr.Checkbox(label="Naya Bold Hook Lagayein (ON / OFF)", value=True)
                hook_source = gr.Radio(["Auto-Generate from Video", "Custom Hook Text"], value="Custom Hook Text", label="Hook Source")
                txt_custom_hook = gr.Textbox(label="Bold Hook Likhein", placeholder="E.g., 99% LOG YEH GALTI KARTE HAIN!")
                hook_col_choice = gr.Radio(["White", "Black"], value="White", label="Hook Text Color")
            with gr.Accordion("✍️ 4. AI Viral Caption Generator", open=True):
                chk_caption = gr.Checkbox(label="Strong Viral Caption Dein (ON / OFF)", value=True)
            run_btn = gr.Button("🚀 Start Processing", variant="primary", size="lg")
        with gr.Column(scale=1):
            video_out = gr.Video(label="Final Edited Video")
            caption_out = gr.Textbox(label="🔥 Viral Post Caption & Hashtags", lines=7)
            status_box = gr.Textbox(label="Live Logs", lines=8)

    run_btn.click(
        fn=process_video,
        inputs=[
            video_in, chk_bg, bg_choice, chk_shorts, sld_duration,
            chk_rem_hook, old_hook_pos, sld_crop_pct,
            chk_new_hook, hook_source, txt_custom_hook, hook_col_choice, chk_caption
        ],
        outputs=[video_out, caption_out, status_box]
    )

if __name__ == "__main__":
    demo.launch(share=True)
