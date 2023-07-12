import os

from pyannote.audio import Inference, Pipeline
import torch
from transformers import WavLMConfig, WavLMModel
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from pyannote.audio.models.segmentation.ecapa_tdnn import ECAPA_TDNN_SMALL
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.models.segmentation import PyanNetWavLM

# load weight

path_weight_segmentation = '../../wavlm_lstm_segmentation.epoch_007.ckpt'
weight_segmentation = torch.load(path_weight_segmentation, map_location=torch.device('cpu'))
# Init segmentation

segmentation = PyanNetWavLM(WavLMModel(WavLMConfig()), use_weighted_sum=True)
segmentation._specifications = weight_segmentation['pyannote.audio']['specifications']
segmentation.build()
segmentation.load_state_dict(weight_segmentation['state_dict'])
segmentation_inference = Inference(
    segmentation,
    duration=5,
    step=0.5,
    skip_aggregation=True,
    batch_size=4,
)
# Init def embedding model

embedding = "microsoft/wavlm-base-plus-sv"
embedding_model = PretrainedSpeakerEmbedding(
    embedding, use_auth_token=os.environ['TOKEN_HUGGINFACE']
)
# Change embedding_model
# https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/models/ecapa_tdnn.py

embedding_model.feature_extractor_ = segmentation.sincnet
embedding_model.feature_extractor_.sampling_rate = weight_segmentation['hyper_parameters']['sample_rate']
embedding_model.model_ = ECAPA_TDNN_SMALL(feat_dim=768, emb_dim=256, )

# Init def pipeline

default_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                            use_auth_token=os.environ['TOKEN_HUGGINFACE'])
#  change default_pipeline

default_pipeline._segmentation = segmentation_inference
default_pipeline._embedding = embedding_model

# inference

default_pipeline("../tests/data/dev00.wav")
