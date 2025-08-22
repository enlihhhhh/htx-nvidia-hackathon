import flask
import base64
import tempfile
import traceback
from flask import Flask, Response, stream_with_context, request
import soundfile as sf
import torch
from kimia_infer.api.kimia import KimiAudio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu" 
# kimi_model_path = os.path.join("/home/htx-team3/", "models/Kimi-Audio-7B-Instruct")
kimi_model_path = "/home/htx-team3/Kimi-Audio/output/Kimi-Audio-7B-Trial-v2"
repo_id = "moonshotai/Kimi-Audio-7B-Instruct"
print("Using model_path:", kimi_model_path)


class KimiChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 model_path=kimi_model_path, device=device) -> None:
        server = Flask(__name__)

        try:
            self.client = KimiAudio(
            model_path=model_path,
            load_detokenizer=True,
            )
        except Exception as e:
            print(f"Failed to load model from local path '{model_path}': {e}")
            print("Attempting to load model from Hugging Face Hub...")
            self.client = KimiAudio(
            model_path=repo_id,
            load_detokenizer=True,
            )

        server.route("/chat", methods=["POST"])(self.chat)

        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    def chat(self) -> Response:
        req_data = flask.request.get_json()
        try:
            # Decode audio
            data_buf = req_data["audio"].encode("utf-8")
            data_buf = base64.b64decode(data_buf)

            citations_text = req_data["citation"]

            print("RAG Citations", citations_text)

            # Save to temp .wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data_buf)
                audio_path = f.name

            # Prepare input for Kimi
            messages = [
                {
                    "role": "user",
                    "message_type": "text",
                    "content": f"You are a professional helpful assistant, helping your user answer queries based on the information you know.\nAnswer the spoken query to the best of your abilities in just ONE sentence. If context is provided, use the pieces of retrieved context to answer the question. Try to understand the speaker's emotion, empathize with it and then reply to the question. If the user asks you a question you do not know the answer to, please acknowledge that you do not have an answer.\nSTRICTLY answer to your user in English ONLY. \n\n CONTEXT:{citations_text}\n",
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio_path, 
                }
            ]
            
            # Sampling parameters

            # For fine-tuned model
            sampling_params = {
                "audio_temperature": 0.1,
                "audio_top_k": 10,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
                "max_new_tokens": 256
            }

            # For base model
            # sampling_params = {
            #     "audio_temperature": 0.8,
            #     "audio_top_k": 10,
            #     "text_temperature": 0.0,
            #     "text_top_k": 5,
            #     "audio_repetition_penalty": 1.5,
            #     "audio_repetition_window_size": 64,
            #     "text_repetition_penalty": 1.5,
            #     "text_repetition_window_size": 16,
            #     "max_new_tokens": 256,
            # }

            output_wav, output_text = self.client.generate(
                messages,
                output_type="both",  # or "both" if you also want `output_text`
                **sampling_params
            )

            print("Kimi Output", output_text)

            # Stream output_wav as audio
            def generate_audio_stream():
                buffer = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(buffer.name, output_wav.squeeze().cpu().numpy(), 24000)
                with open(buffer.name, "rb") as f:
                    while chunk := f.read(4096):
                        yield chunk

            return Response(stream_with_context(generate_audio_stream()), mimetype="audio/wav")

        except Exception as e:
            print(traceback.format_exc())
            return Response("Error occurred", status=500)


def create_app():
    server = KimiChatServer(run_app=False)
    return server.server


def serve(ip='0.0.0.0', port=60808, device='cuda:0'):
    KimiChatServer(ip=ip, port=port, run_app=True, device=device)


if __name__ == "__main__":
    import fire
    fire.Fire(serve)
