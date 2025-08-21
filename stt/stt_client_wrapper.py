import os 
from pathlib import Path
from itertools import groupby
import grpc
import riva.client
import yaml
from types import SimpleNamespace
from dotenv import load_dotenv

# Load config as args
CONFIG_FILE="stt_config_api.yaml"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, CONFIG_FILE)

with open(CONFIG_PATH) as f:
    data = yaml.safe_load(f)

args = SimpleNamespace(**data)

class ParakeetSTTClient:
    def __init__(
            self,
            input_file: str,
            api_key: str,
            args: SimpleNamespace = args,
            ):
        self.input_file = input_file
        self.api_key = api_key
        self.args = args
        self._process_args()

        self.options = [('grpc.max_receive_message_length', args.max_message_length), ('grpc.max_send_message_length', args.max_message_length)]

        self.auth = riva.client.Auth(
            ssl_root_cert=args.ssl_root_cert,
            ssl_client_cert=args.ssl_client_cert,
            ssl_client_key=args.ssl_client_key,
            use_ssl=args.use_ssl,
            uri=args.server,
            metadata_args=args.metadata,
            options=self.options
            )
        
        self.asr_service = riva.client.ASRService(self.auth)
    
    def _process_args(self):
        if (self.input_file is None and not self.args.list_models) or \
            (self.input_file is not None and self.args.list_models):
            raise ValueError("You must specify exactly only one of 'input_file' or 'list_models'")
        
        if self.input_file:
            self.args.input_file = Path(self.input_file).expanduser()

        if self.args.metadata:
            self.args.metadata = [
                ["function-id", "d3fe9151-442b-4204-a70d-5fcc597fd610"],
                ["authorization", f"Bearer {self.api_key}"],
                ]
        else:
            self.args.metadata = None
    
    def _write_seglst(words, seglst_output_file):
    # Sort words by start_time to ensure chronological order
        sorted_words = sorted(words, key=lambda word: word.start_time)
        
        seglst = []
        for speaker_tag, group in groupby(sorted_words, key=lambda word: word.speaker_tag):
            group_words = list(group)
            seg = {
                "session_id": seglst_output_file,
                "words": " ".join(word.word for word in group_words),
                "start_time": str(group_words[0].start_time / 1000),
                "end_time": str(group_words[-1].end_time / 1000),
                "speaker": f"speaker{int(speaker_tag) + 1}",
            }
            seglst.append(seg)
        
        return seglst
    
    def list_models(self):
        if args.list_models:
            asr_models = dict()
            config_response = self.asr_service.stub.GetRivaSpeechRecognitionConfig(riva.client.proto.riva_asr_pb2.RivaSpeechRecognitionConfigRequest())
            for model_config in config_response.model_config:
                if model_config.parameters["type"] == "offline":
                    language_code = model_config.parameters['language_code']
                    model = {"model": [model_config.model_name]}
                    if language_code in asr_models:
                        asr_models[language_code].append(model)
                    else:
                        asr_models[language_code] = [model]

            print("Available ASR models")
            asr_models = dict(sorted(asr_models.items()))
            print(asr_models)
            return
        else:
            raise ValueError("Must set list_models to true in yaml")

    def transcribe(self):
        if not os.path.isfile(self.args.input_file):
            raise FileNotFoundError(f"Invalid input file path: {self.args.input_file}")

        config = riva.client.RecognitionConfig(
            language_code=args.language_code,
            max_alternatives=args.max_alternatives,
            profanity_filter=args.profanity_filter,
            enable_automatic_punctuation=args.automatic_punctuation,
            verbatim_transcripts=not args.no_verbatim_transcripts,
            enable_word_time_offsets=args.word_time_offsets or args.speaker_diarization,
        )
        riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
        riva.client.add_speaker_diarization_to_config(config, args.speaker_diarization, args.diarization_max_speakers)
        riva.client.add_endpoint_parameters_to_config(
            config,
            args.start_history,
            args.start_threshold,
            args.stop_history,
            args.stop_history_eou,
            args.stop_threshold,
            args.stop_threshold_eou
        )
        riva.client.add_custom_configuration_to_config(
            config,
            args.custom_configuration
        )
        with args.input_file.open('rb') as fh:
            data = fh.read()
        try:
            seglst_output_file = None
            if args.output_seglst:
                seglst_output_file = os.path.basename(args.input_file).split(".")[0]
            
            speaker_diarization=args.speaker_diarization
            # Original code on commented out line below: 
            #riva.client.print_offline(response=asr_service.offline_recognize(data, config), speaker_diarization=args.speaker_diarization, seglst_output_file=seglst_output_file)
            response = self.asr_service.offline_recognize(data, config)
            if len(response.results) > 0 and len(response.results[0].alternatives) > 0:
                final_transcript = ""
                words = []
                for res in response.results:
                    final_transcript += res.alternatives[0].transcript
                    if speaker_diarization:
                        for word_info in res.alternatives[0].words:
                            words.append(word_info)

                # TODO <Ros> : Currently does not return anything on diarized content
                if speaker_diarization and len(words) > 0 and seglst_output_file is not None:
                    self._write_seglst(words, seglst_output_file)

                return final_transcript

        except grpc.RpcError as e:
            print(e.details())
            
# Test usage
if __name__ == "__main__": 
    # cd to root folder, then run $ uv run stt/<name_of_this_script.py>
    load_dotenv()
    api_key = os.environ.get("RIVA_API_KEY")  

    print(api_key)

    audio = "monologue_1min.wav"
    
    client = ParakeetSTTClient(
        input_file=audio,
        api_key=api_key
    )

    transcription = client.transcribe()
    print(transcription)
