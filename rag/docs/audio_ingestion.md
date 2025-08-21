<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Enable audio ingestion support
Enabling audio ingestion support allows the system to process and transcribe audio files (.mp3 and .wav) during document ingestion. This enables better search and retrieval capabilities for audio content in your documents.

Once you have followed [steps in quick start guide](./quickstart.md#deploy-with-docker-compose) to launch the blueprint, to enable audio ingestion support, follow these steps:

## Using on-prem audio transcription model

### Docker Compose Flow

1. Deploy the audio transcription model on-prem. You need a GPU to deploy this model. For a list of supported GPUs, see [NVIDIA Riva ASR Support Matrix](https://docs.nvidia.com/nim/riva/asr/latest/support-matrix.html#gpus-supported).
   ```bash
   USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile audio up -d
   ```

2. Make sure the audio container is up and running
   ```bash
   docker ps --filter "name=audio" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*
   ```output
   NAMES                                   STATUS
   compose-audio-1                         Up 5 minutes (healthy)
   ```

3. The ingestor-server is already configured to handle audio files. You can now ingest audio files (.mp3 or .wav) using the ingestion API as shown in the [ingestion API usage notebook](../notebooks/ingestion_api_usage.ipynb).

   Example usage with the ingestion API:
   ```python
   FILEPATHS = [
       '../data/audio/sample.mp3',
       '../data/audio/sample.wav'
   ]

   await upload_documents(collection_name="audio_data")
   ```

> [!Note]
> The audio transcription service requires GPU resources. Make sure you have sufficient GPU resources available before enabling this feature.

### Customizing GPU Usage for Audio Service (Optional)

By default, the `audio` service uses GPU ID 0. You can customize which GPU to use by setting the `AUDIO_MS_GPU_ID` environment variable before starting the service:

```bash
export AUDIO_MS_GPU_ID=3  # Use GPU 3 instead of GPU 0
USERID=$(id -u) docker compose -f deploy/compose/nims.yaml --profile audio up -d
```

Alternatively, you can modify the `nims.yaml` file directly to change the GPU assignment:

```yaml
# In deploy/compose/nims.yaml, locate the audio service and modify:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["${AUDIO_MS_GPU_ID:-0}"]  # Change 0 to your desired GPU ID
          capabilities: [gpu]
```

> [!Note]
> Ensure the specified GPU is available and has sufficient memory for the audio transcription model. The Riva ASR model typically requires at least 8GB of GPU memory.

### Helm Flow

If you're using Helm for deployment, follow these steps to enable audio ingestion:

1. Enable Riva NIM in your main `values.yaml` file, enable the riva-nim deployment by setting:
   ```yaml
   ingestor-server:
     nv-ingest:
       riva-nim:
         deployed: true
   ```

2. Ensure that audio extraction dependencies are installed in nv-ingest by verifying this setting in your `values.yaml`:

   ```yaml
   ingestor-server:
     nv-ingest:
       envVars:
         INSTALL_AUDIO_EXTRACTION_DEPS: "true"
   ```

3. Apply the updated Helm chart by running the following code.

   ```bash
   helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.2.0.tgz \
   --username '$oauthtoken' \
   --password "${NGC_API_KEY}" \
   --set imagePullSecret.password=$NGC_API_KEY \
   --set ngcApiSecret.password=$NGC_API_KEY \
   -f rag-server/values.yaml
   ```

4. Verify that the riva-nim pod is running:
   ```bash
   kubectl get pods -n rag | grep riva-nim
   ```
   Output:
   ```bash
      nv-ingest-riva-nim-6578f4579f-4q75k                         1/1     Running   0             3m29s
   ```
   ```bash
   kubectl get svc -n rag | grep riva-nim
   ```
   Output:
   ```bash
      nv-ingest-riva-nim                  ClusterIP   10.103.184.78    <none>        9000/TCP,50051/TCP   4m27s
   ```

> [!Important]
> When using Helm deployment, the Riva NIM service requires an additional H100 or B200 GPU making the total GPU requirement to 9xH100 without MIG slicing.

## Limitations

When ingesting audio files, please note the following limitations:

- Audio segmentation is not supported. Each audio file is processed as a single unit.
- Audio ingestion is not recommended for larger files as each audio file transcript forms a single chunk. For optimal performance, it is recommended to use audio files with less than 2 minutes duration each. Large audio file transcripts may exceed the generation LLM context window limits.
- The chunk size and overlap settings in the ingestion API's `split_options` parameter do not affect audio file processing.
