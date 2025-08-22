
import os
import argparse
from pathlib import Path
from itertools import groupby
import grpc
import riva.client
from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline file transcription via Riva AI Services. \"Offline\" means that entire audio "
        "content of `--input-file` is sent in one request and then a transcript for whole file recieved in "
        "one response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=Path, help="A path to a local file to transcribe.")
    group.add_argument("--list-models", action="store_true", help="List available models.")
    parser.add_argument("--output-seglst", action="store_true", help="Output seglst file for speaker diarization.")

    parser = add_connection_argparse_parameters(parser)
    parser = add_asr_config_argparse_parameters(parser, max_alternatives=True, profanity_filter=True, word_time_offsets=True)
    args = parser.parse_args()
    if args.input_file:
        args.input_file = args.input_file.expanduser()

    return args

def write_seglst(words, seglst_output_file):
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

def main() -> None:
    args = parse_args()

    options = [('grpc.max_receive_message_length', args.max_message_length), ('grpc.max_send_message_length', args.max_message_length)]
    auth = riva.client.Auth(
        ssl_root_cert=args.ssl_root_cert,
        ssl_client_cert=args.ssl_client_cert,
        ssl_client_key=args.ssl_client_key,
        use_ssl=args.use_ssl,
        uri=args.server,
        metadata_args=args.metadata,
        options=options)
    asr_service = riva.client.ASRService(auth)

    if args.list_models:
        asr_models = dict()
        config_response = asr_service.stub.GetRivaSpeechRecognitionConfig(riva.client.proto.riva_asr_pb2.RivaSpeechRecognitionConfigRequest())
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

    if not os.path.isfile(args.input_file):
        print(f"Invalid input file path: {args.input_file}")
        return

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
        response = asr_service.offline_recognize(data, config)
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
                write_seglst(words, seglst_output_file)

            print(final_transcript)

    except grpc.RpcError as e:
        print(e.details())


if __name__ == "__main__":
    main()
