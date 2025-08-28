import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import assemblyai as aai
import tempfile
from src.ai_component.config import Config


class AudioTranscriber:
    def __init__(self ):
        aai.settings.api_key = Config.assemblyai_api_key
        self.config = aai.TranscriptionConfig(
            speech_model =  aai.SpeechModel.best,
            language_detection = True,
            punctuate= True,
            format_text = True
        )
        self.transcriber = aai.Transcriber(config = self.config)

    def transcriber_bytes(self, audio_bytes):
        """Trnascribe audio bytes to text"""
        try:
            if not audio_bytes or len(audio_bytes) == 0:
                return {
                    "success": False,
                    "error": "No audio data provided"
                }
            ## save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name

            ## checking file size
            file_size = os.path.getsize(tmp_file_path)
            if file_size < 1000:  ## less than 1 KB
                os.unlink(tmp_file_path)
                return {
                    "success": False,
                    "error": "Audio file is too small"
                }
            
            ## transcribe
            transcript = self.transcriber.transcribe(tmp_file_path)

            ## cleanup
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            if transcript.status == aai.TranscriptStatus.error:
                return {"success": False, "error": f"Transcription failed: {transcript.error}"}
            elif transcript.status == aai.TranscriptStatus.completed:
                if transcript.text and len(transcript.text.strip()) > 0:
                    return {"success": True, "text": transcript.text.strip()}
                else:
                    return {"success": False, "error": "No speech detected. Please speak clearly and try again."}
            else:
                return {"success": False, "error": f"Unexpected status: {transcript.status}"}
                
        except Exception as e:
            return {"success": False, "error": f"Transcription error: {str(e)}"}
        

# transcriber = AudioTranscriber()
# result = transcriber.transcribe_bytes(audio_bytes)
# if result["success"]:
#     print(result["text"])
# else:
#     print(result["error"]) 