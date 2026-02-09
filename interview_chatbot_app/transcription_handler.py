import queue
import threading
import json
import logging
import av
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    from streamlit_webrtc import AudioProcessorBase
except ImportError:
    # Fallback if streamlit-webrtc is not installed (e.g. missing system deps)
    class AudioProcessorBase:
        def callback(self, frame):
            pass

from deepgram import DeepgramClient
# Try to import EventType, handle likely location
try:
    from deepgram.core.events import EventType
except ImportError:
    # If not found there, define simple string constants or try another path
    # But based on file inspection it should be there.
    class EventType:
        OPEN = "open"
        MESSAGE = "message"
        CLOSE = "close"
        ERROR = "error"

def transcribe_audio_deepgram(audio_bytes, api_key):
    """Uses Deepgram for ultra-fast transcription (v5 SDK syntax/Fern)."""
    if not api_key:
        return "Error: Deepgram API Key is missing."
    
    try:
        dg_client = DeepgramClient(api_key=api_key)
        
        # REST API for file transcription
        # Note: In Fern SDK, options might need to be passed differently or as specific types.
        # We try passing as kwargs which is standard in many generated SDKs.
        # If this fails, we might need to check the specific signature.
        
        # Checking imports in listen/v1/__init__.py showed MediaTranscribeRequest... 
        # But high level method usually abstracts this.
        
        response = dg_client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-2",
            smart_format=True,
            language="en-US"
        )
        
        # Response structure
        # response should be MediaTranscribeResponse
        # results -> channels -> [0] -> alternatives -> [0] -> transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript.strip() if transcript else "No transcript generated."
    except Exception as e:
        # Fallback for older SDK or different signature if needed, but reporting error first
        return f"Deepgram Error: {e}"

class AudioProcessor(AudioProcessorBase):
    def __init__(self, api_key):
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.api_key = api_key
        self.is_running = True
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        self.thread = threading.Thread(target=self._deepgram_thread, daemon=True)
        self.thread.start()

    def recv(self, frame):
        return self.recv_audio(frame)

    def recv_audio(self, frame):
        try:
            # Resample to 16kHz mono s16le
            resampled_frames = self.resampler.resample(frame)
            for resampled_frame in resampled_frames:
                audio_data = resampled_frame.to_ndarray().tobytes()
                self.audio_queue.put(audio_data)
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
        return frame

    def _deepgram_thread(self):
        """Websocket thread for Deepgram Live (Fern SDK)."""
        logger.info("Deepgram thread started.")
        try:
            client = DeepgramClient(api_key=self.api_key)
            
            # Helper to run the listener in a separate thread
            def start_listener(socket):
                socket.start_listening()

            # Connect using context manager
            # Note: All params must be strings as per SDK definition file inspection
            with client.listen.v1.connect(
                model="nova-2",
                language="en-US",
                smart_format="true",
                interim_results="false",
                encoding="linear16",
                sample_rate="16000",
                endpointing="300"
            ) as socket:
                
                # Define callback
                def on_message(result, **kwargs):
                    # result is the parsed object
                    try:
                        # Depending on event type (metadata vs results)
                        # We look for channel.alternatives[0].transcript
                        if hasattr(result, 'channel'):
                            transcript = result.channel.alternatives[0].transcript
                            if transcript:
                                self.transcript_queue.put(transcript)
                    except Exception:
                        pass # Ignore parsing errors or metadata events

                # Register callback
                socket.on(EventType.OPEN, lambda *args: logger.info("Deepgram Socket Open"))
                socket.on(EventType.CLOSE, lambda *args: logger.info("Deepgram Socket Closed"))
                socket.on(EventType.ERROR, lambda error, **kwargs: logger.error(f"Deepgram Socket Error: {error}"))
                socket.on(EventType.MESSAGE, on_message)
                
                # Start listener thread (non-blocking for us, but blocks that thread)
                listener_thread = threading.Thread(target=start_listener, args=(socket,), daemon=True)
                listener_thread.start()

                # Send audio loop
                while self.is_running:
                    try:
                        data = self.audio_queue.get(timeout=1)
                        # Use send_media(bytes)
                        # logger.info(f"Sending {len(data)} bytes to Deepgram") # Uncomment for verbose debugging
                        if len(data) > 0:
                            socket.send_media(data)
                        else:
                            logger.warning("Empty audio chunk skipped")
                    except queue.Empty:
                        pass
                
                # When loop ends (is_running=False), context manager exit will close socket
        except Exception as e:
            self.transcript_queue.put(f"Deepgram Error: {e}")
            logger.error(f"Deepgram Thread Exiting with error: {e}")

    def on_ended(self):
        self.is_running = False
