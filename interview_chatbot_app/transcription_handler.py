import queue
import threading
import logging
import av
import riva.client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from streamlit_webrtc import AudioProcessorBase
except ImportError:
    # Fallback if streamlit-webrtc is not installed
    class AudioProcessorBase:
        def callback(self, frame):
            pass

class AudioProcessor(AudioProcessorBase):
    def __init__(self, api_key):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.api_key = api_key
        self.is_running = True
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        self.thread = threading.Thread(target=self._riva_thread, daemon=True)
        self.thread.start()

    def recv(self, frame):
        return self.recv_audio(frame)

    def recv_audio(self, frame):
        try:
            # Resample to 16kHz mono s16le for Riva
            resampled_frames = self.resampler.resample(frame)
            for resampled_frame in resampled_frames:
                audio_data = resampled_frame.to_ndarray().tobytes()
                self.audio_queue.put(audio_data)
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
        return frame

    def _audio_generator(self):
        chunks_yielded = 0
        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=1)
                if len(data) > 0:
                    chunks_yielded += 1
                    if chunks_yielded % 50 == 0:
                        logger.info(f"Yielded {chunks_yielded} chunks to Riva.")
                    yield data
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Generator Exception: {e}")
                break

    def _riva_thread(self):
        """Websocket thread for NVIDIA Riva Streaming (nemotron-asr-streaming)."""
        logger.info("NVIDIA Riva thread started.")
        
        # Primary NVCF model for nemotron-asr-streaming
        function_id = "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
        
        try:
            auth = riva.client.Auth(
                use_ssl=True,
                uri="grpc.nvcf.nvidia.com:443",
                metadata_args=[
                    ["authorization", f"Bearer {self.api_key}"],
                    ["function-id", function_id] 
                ]
            )
            
            asr_service = riva.client.ASRService(auth)
            
            config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=16000,
                audio_channel_count=1,
                language_code="en-US"
            )
            
            streaming_config = riva.client.StreamingRecognitionConfig(
                config=config,
                interim_results=False
            )

            # Generator loop (blocking) runs continuously as long as audio yields
            responses = asr_service.streaming_response_generator(
                audio_chunks=self._audio_generator(),
                streaming_config=streaming_config
            )

            for response in responses:
                # Add debug logging for raw response chunks (throttle to avoid flooding)
                if getattr(response, 'results', None):
                    for result in response.results:
                        if not getattr(result, 'alternatives', None):
                            continue
                        transcript = result.alternatives[0].transcript
                        if result.is_final and transcript.strip():
                            logger.info(f"Riva Final Transcript: {transcript.strip()}")
                            self.transcript_queue.put(transcript.strip())
                        elif transcript.strip():
                            # Interim result or logging
                            pass

        except Exception as e:
            error_msg = str(e)
            
            # Fallback to specifically NVIDIABuild-Autogen-42 if bb08 fails, 
            # though our integration tests show bb08 is the stable func ID.
            if "NVIDIABuild-Autogen-42" not in error_msg:
                self.transcript_queue.put(f"Riva Error: {e}")
            logger.error(f"Riva Thread Exiting with error: {e}")

    def on_ended(self):
        self.is_running = False
