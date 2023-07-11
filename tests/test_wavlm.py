import os

from pyannote.audio import Inference, Pipeline
import torch
from transformers import WavLMConfig, WavLMModel
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.models.segmentation import PyanNetWavLM

path_weight_segmentation = '../../wavlm_lstm_segmentation.epoch_007.ckpt'
weight_segmentation = torch.load(path_weight_segmentation, map_location=torch.device('cpu'))
segmentation = PyanNetWavLM(WavLMModel(WavLMConfig()), use_weighted_sum=True)
segmentation._specifications = weight_segmentation['pyannote.audio']['specifications']
segmentation.build()
segmentation.load_state_dict(weight_segmentation['state_dict'])

default_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                            use_auth_token=os.environ['TOKEN_HUGGINFACE'])
default_pipeline._segmentation = Inference(
    segmentation,
    duration=5,
    step=0.5,
    skip_aggregation=True,
    batch_size=4,
)

default_pipeline("data/dev00.wav")
# default_pipeline("data/empty.wav")
