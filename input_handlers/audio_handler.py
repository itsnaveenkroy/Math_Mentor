"""
Audio input handler using SpeechRecognition library.
Falls back to Google's free speech-to-text API - no keys needed.
Then runs basic math-term normalization on the transcript.
"""

import io
import re
import struct
import tempfile
import os


def _import_sr():
    """Lazy import so we get a clear error at call time, not module load."""
    try:
        import speech_recognition as sr
        return sr
    except ImportError:
        return None


def _import_pydub():
    try:
        from pydub import AudioSegment
        return AudioSegment
    except ImportError:
        return None


# spoken math phrases -> symbolic notation
MATH_PHRASE_MAP = [
    (r'\bsquare root of\b', 'sqrt'),
    (r'\bsquare root\b', 'sqrt'),
    (r'\bsquared\b', '^2'),
    (r'\bcubed\b', '^3'),
    (r'\bto the power of\b', '^'),
    (r'\braised to\b', '^'),
    (r'\btimes\b', '*'),
    (r'\bmultiplied by\b', '*'),
    (r'\bdivided by\b', '/'),
    (r'\bover\b', '/'),
    (r'\bplus\b', '+'),
    (r'\bminus\b', '-'),
    (r'\bequals?\b', '='),
    (r'\bis equal to\b', '='),
    (r'\bgreater than or equal\b', '>='),
    (r'\bless than or equal\b', '<='),
    (r'\bgreater than\b', '>'),
    (r'\bless than\b', '<'),
    (r'\bopen paren\b', '('),
    (r'\bclose paren\b', ')'),
    (r'\bpi\b', 'pi'),
    (r'\binfinity\b', 'inf'),
    (r'\bintegral of\b', 'integrate'),
    (r'\bderivative of\b', 'diff'),
    (r'\blimit of\b', 'limit'),
    (r'\bsine\b', 'sin'),
    (r'\bcosine\b', 'cos'),
    (r'\btangent\b', 'tan'),
    (r'\blog of\b', 'log'),
    (r'\bnatural log\b', 'ln'),
]


def _normalize_math_phrases(text):
    """Convert spoken math to something more symbolic."""
    result = text.lower()
    for pattern, replacement in MATH_PHRASE_MAP:
        result = re.sub(pattern, replacement, result)
    return result.strip()


def _apply_learned_corrections(text):
    """Apply corrections from memory (learned from past user edits)."""
    try:
        from memory.memory_store import get_corrections
        corrections = get_corrections(source="audio", n=50)
        for c in corrections:
            original = c.get("original", "")
            corrected = c.get("corrected", "")
            if original and corrected and original in text:
                text = text.replace(original, corrected)
    except Exception:
        pass  # memory not available, skip
    return text


def _write_temp(data, suffix=".wav"):
    """Write bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def _detect_audio_format(audio_bytes):
    """
    Detect audio format from magic bytes.
    Returns a short extension string: '.wav', '.webm', '.ogg', '.mp3', '.mp4', or None.
    """
    if len(audio_bytes) < 12:
        return None
    if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
        return '.wav'
    if audio_bytes[:4] == b'OggS':
        return '.ogg'
    # WebM / Matroska starts with EBML header: 0x1A 0x45 0xDF 0xA3
    if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
        return '.webm'
    # MP3: ID3 tag or MPEG sync word
    if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb' or audio_bytes[:2] == b'\xff\xf3':
        return '.mp3'
    # MP4/M4A: check for 'ftyp' box
    if audio_bytes[4:8] == b'ftyp':
        return '.mp4'
    # FLAC
    if audio_bytes[:4] == b'fLaC':
        return '.flac'
    return None


def _normalize_wav_bytes(audio_bytes):
    """
    Convert any WAV to standard 16-bit mono PCM WAV by parsing the raw bytes.

    Python's stdlib wave module CANNOT open float32 WAV (format tag 3),
    which is exactly what browsers produce via st.audio_input.
    So we parse the RIFF/WAV header manually.

    Returns normalized WAV bytes, or None if not a WAV or parsing fails.
    """
    try:
        data = audio_bytes
        if len(data) < 44:
            return None

        # parse RIFF header
        riff_id = data[0:4]
        if riff_id != b"RIFF":
            return None
        wave_id = data[8:12]
        if wave_id != b"WAVE":
            return None

        # find fmt and data chunks
        pos = 12
        fmt_tag = None
        channels = None
        sample_rate = None
        bits_per_sample = None
        raw_samples = None

        while pos < len(data) - 8:
            chunk_id = data[pos : pos + 4]
            chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
            chunk_data = data[pos + 8 : pos + 8 + chunk_size]

            if chunk_id == b"fmt ":
                fmt_tag = struct.unpack_from("<H", chunk_data, 0)[0]
                channels = struct.unpack_from("<H", chunk_data, 2)[0]
                sample_rate = struct.unpack_from("<I", chunk_data, 4)[0]
                bits_per_sample = struct.unpack_from("<H", chunk_data, 14)[0]
            elif chunk_id == b"data":
                raw_samples = chunk_data

            pos += 8 + chunk_size
            # chunks are word-aligned
            if chunk_size % 2 == 1:
                pos += 1

        if fmt_tag is None or raw_samples is None or channels is None:
            return None

        print(f"[audio_handler] WAV header: fmt_tag={fmt_tag}, ch={channels}, "
              f"rate={sample_rate}, bits={bits_per_sample}, "
              f"data_bytes={len(raw_samples)}")

        # fmt_tag 1 = PCM, 3 = IEEE float
        # if already 16-bit PCM mono at <=16kHz, return as-is
        if fmt_tag == 1 and bits_per_sample == 16 and channels == 1 and sample_rate <= 16000:
            return audio_bytes

        # convert samples to int16
        n_samples = len(raw_samples) // (bits_per_sample // 8)

        if fmt_tag == 3 and bits_per_sample == 32:
            # IEEE float32 → int16
            float_samples = struct.unpack(f"<{n_samples}f", raw_samples)
            int16 = [int(max(-1.0, min(1.0, s)) * 32767) for s in float_samples]
        elif fmt_tag == 1 and bits_per_sample == 32:
            # int32 → int16
            int32 = struct.unpack(f"<{n_samples}i", raw_samples)
            int16 = [s >> 16 for s in int32]
        elif fmt_tag == 1 and bits_per_sample == 24:
            # int24 → int16 (unpack 3 bytes per sample)
            int16 = []
            for i in range(0, len(raw_samples), 3):
                if i + 2 < len(raw_samples):
                    val = raw_samples[i] | (raw_samples[i + 1] << 8) | (raw_samples[i + 2] << 16)
                    if val >= 0x800000:
                        val -= 0x1000000
                    int16.append(val >> 8)
            n_samples = len(int16)
        elif fmt_tag == 1 and bits_per_sample == 16:
            int16 = list(struct.unpack(f"<{n_samples}h", raw_samples))
        elif fmt_tag == 1 and bits_per_sample == 8:
            # unsigned 8-bit → int16
            int16 = [(s - 128) * 256 for s in raw_samples]
            n_samples = len(int16)
        elif fmt_tag == 3 and bits_per_sample == 64:
            # float64 → int16
            n_samples = len(raw_samples) // 8
            float_samples = struct.unpack(f"<{n_samples}d", raw_samples)
            int16 = [int(max(-1.0, min(1.0, s)) * 32767) for s in float_samples]
        else:
            return None  # unsupported format

        # mix to mono if stereo
        if channels >= 2:
            mono = []
            for i in range(0, len(int16) - channels + 1, channels):
                avg = sum(int16[i : i + channels]) // channels
                mono.append(avg)
            int16 = mono

        # build a clean PCM WAV
        pcm_data = struct.pack(f"<{len(int16)}h", *int16)
        return _build_wav(pcm_data, sample_rate, 1, 2)

    except Exception:
        return None


def _build_wav(pcm_data, sample_rate, channels, sample_width):
    """Build a standard WAV file from raw PCM data."""
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,              # fmt chunk size
        1,               # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,  # bits per sample
        b"data",
        data_size,
    )
    return header + pcm_data


def _try_read_audio(sr, recognizer, audio_bytes, ext):
    """
    Try to read audio into a speech_recognition AudioData object.
    Auto-detects the real format from magic bytes (browsers lie about format).
    """
    # Auto-detect real format from magic bytes — critical because
    # st.audio_input() often returns WebM/OGG even though we label it .wav
    detected = _detect_audio_format(audio_bytes)
    real_ext = detected or ext
    print(f"[audio_handler] filename ext={ext}, detected format={detected}, "
          f"first 12 bytes={audio_bytes[:12].hex()}, size={len(audio_bytes)}")

    # --- Path A: actual WAV → normalize + resample to 16kHz ---
    if real_ext == ".wav":
        normalized = _normalize_wav_bytes(audio_bytes)
        if normalized is not None:
            # Resample to 16kHz mono via pydub+ffmpeg — critical because
            # browsers record at 48kHz but Google's free API needs ≤16kHz
            AudioSegment = _import_pydub()
            if AudioSegment is not None:
                try:
                    seg = AudioSegment.from_file(io.BytesIO(normalized), format="wav")
                    seg = seg.set_sample_width(2).set_channels(1).set_frame_rate(16000)
                    wav_buf = io.BytesIO()
                    seg.export(wav_buf, format="wav")
                    resampled = wav_buf.getvalue()
                    print(f"[audio_handler] resampled WAV: {len(normalized)}→{len(resampled)} bytes (16kHz)")
                    normalized = resampled
                except Exception as e:
                    print(f"[audio_handler] resample failed, using original rate: {e}")

            tmp_path = _write_temp(normalized, suffix=".wav")
            try:
                with sr.AudioFile(tmp_path) as source:
                    return recognizer.record(source)
            except Exception:
                pass
            finally:
                os.unlink(tmp_path)

    # --- Path B: non-WAV (WebM, OGG, MP3, etc.) → convert via pydub → WAV ---
    AudioSegment = _import_pydub()
    if AudioSegment is not None and real_ext in (".webm", ".ogg", ".mp3", ".mp4", ".m4a", ".flac"):
        fmt_map = {".mp3": "mp3", ".m4a": "mp4", ".mp4": "mp4", ".ogg": "ogg",
                   ".flac": "flac", ".webm": "webm", ".wav": "wav"}
        fmt = fmt_map.get(real_ext, real_ext.lstrip("."))
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
            seg = seg.set_sample_width(2).set_channels(1).set_frame_rate(16000)
            wav_buf = io.BytesIO()
            seg.export(wav_buf, format="wav")
            wav_data = wav_buf.getvalue()
            print(f"[audio_handler] pydub converted {fmt}→wav: {len(wav_data)} bytes")
            tmp_path = _write_temp(wav_data, suffix=".wav")
            try:
                with sr.AudioFile(tmp_path) as source:
                    return recognizer.record(source)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"[audio_handler] pydub conversion failed: {e}")

    # --- Path C: direct read fallback (standard PCM wav, flac, aiff) ---
    tmp_path = _write_temp(audio_bytes, suffix=real_ext)
    try:
        with sr.AudioFile(tmp_path) as source:
            return recognizer.record(source)
    except Exception:
        pass
    finally:
        os.unlink(tmp_path)

    # --- Path D: last resort – pydub with original ext ---
    if AudioSegment is not None:
        try:
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            seg = seg.set_sample_width(2).set_channels(1).set_frame_rate(16000)
            wav_buf = io.BytesIO()
            seg.export(wav_buf, format="wav")
            tmp_path = _write_temp(wav_buf.getvalue(), suffix=".wav")
            try:
                with sr.AudioFile(tmp_path) as source:
                    return recognizer.record(source)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f"[audio_handler] pydub last-resort failed: {e}")

    raise RuntimeError(
        "Could not decode the audio file. Try converting it to a standard "
        "16-bit PCM .wav file (e.g. using Audacity or an online converter)."
    )


def transcribe_audio(audio_bytes, filename="audio.wav"):
    """
    Transcribe audio bytes to text using Google's free speech API.
    Returns dict with text, confidence info, etc.
    """
    sr = _import_sr()
    if sr is None:
        import sys
        return {
            "text": "",
            "confidence": 0.0,
            "needs_hitl": True,
            "error": "speech_recognition not importable",
            "message": f"SpeechRecognition not found in {sys.executable}. Run: {sys.executable} -m pip install SpeechRecognition",
        }

    try:
        ext = os.path.splitext(filename)[1].lower() or ".wav"
        recognizer = sr.Recognizer()

        # _try_read_audio normalizes WAV to 16-bit PCM first (handles browser float32)
        audio = _try_read_audio(sr, recognizer, audio_bytes, ext)

        # Diagnostic: log audio data properties
        print(f"[audio_handler] AudioData: rate={audio.sample_rate}, "
              f"width={audio.sample_width}, "
              f"frame_bytes={len(audio.frame_data)}, "
              f"duration={len(audio.frame_data)/audio.sample_rate/audio.sample_width:.2f}s")

        # Compute RMS energy to check if audio is silent
        import array
        pcm = array.array('h', audio.frame_data)
        if len(pcm) > 0:
            rms = (sum(s * s for s in pcm) / len(pcm)) ** 0.5
            peak = max(abs(s) for s in pcm)
            print(f"[audio_handler] Audio energy: rms={rms:.1f}, peak={peak}, "
                  f"(max possible=32767)")
        else:
            print("[audio_handler] WARNING: audio frame_data is EMPTY")

        # Save debug copy for inspection
        debug_path = os.path.join(tempfile.gettempdir(), "math_mentor_debug.wav")
        debug_wav = audio.get_wav_data()
        with open(debug_path, "wb") as f:
            f.write(debug_wav)
        print(f"[audio_handler] Debug WAV saved to: {debug_path} ({len(debug_wav)} bytes)")

        # Try show_all=True first to see raw Google response
        try:
            full_response = recognizer.recognize_google(audio, show_all=True)
            print(f"[audio_handler] Google raw response: {full_response}")
        except Exception as e:
            print(f"[audio_handler] Google show_all failed: {e}")

        # google's free api - no key needed, decent accuracy
        raw_text = recognizer.recognize_google(audio)

        # clean up math phrasing
        cleaned = _normalize_math_phrases(raw_text)

        # apply learned corrections from past user edits
        cleaned = _apply_learned_corrections(cleaned)

        return {
            "text": cleaned,
            "raw_text": raw_text,
            "confidence": 0.8,  # google api doesn't give us confidence easily
            "needs_hitl": True,  # always let user verify audio
            "error": None,
            "message": "Double-check the transcription, speech-to-text can be iffy with math.",
        }

    except sr.UnknownValueError:
        return {
            "text": "",
            "confidence": 0.0,
            "needs_hitl": True,
            "error": "Could not understand audio",
            "message": "Couldn't make out what was said. Try again or type it manually.",
        }
    except sr.RequestError as e:
        return {
            "text": "",
            "confidence": 0.0,
            "needs_hitl": True,
            "error": str(e),
            "message": f"Speech service error: {e}",
        }
    except Exception as e:
        return {
            "text": "",
            "confidence": 0.0,
            "needs_hitl": True,
            "error": str(e),
            "message": f"Audio processing failed: {e}",
        }
