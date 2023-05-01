import json
import os
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from assistant_server.gesture_generation.anim import bvh, quat
from assistant_server.gesture_generation.anim.txform import \
    xform_orthogonalize_from_xy
from assistant_server.gesture_generation.audio.audio_files import read_wavfile
from assistant_server.gesture_generation.data_pipeline import (
    preprocess_animation, preprocess_audio)
from assistant_server.gesture_generation.helpers import split_by_ratio
from assistant_server.gesture_generation.postprocessing import reset_pose
from assistant_server.gesture_generation.utils import timeit, write_bvh


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class GestureInferenceModelConfig:
    device: str
    data_pipeline_conf: DictConfig
    audio_input_mean: Tensor
    audio_input_std: Tensor
    anim_input_mean: Tensor
    anim_input_std: Tensor
    anim_output_mean: Tensor
    anim_output_std: Tensor
    labels: List[str]
    bones: List[str]
    parents: Tensor
    dt: float
    network_speech_encoder: torch.nn.Module
    network_speech_encoder_script: torch.nn.Module
    network_decoder: torch.nn.Module
    network_decoder_script: torch.nn.Module
    network_style_encoder: Optional[torch.nn.Module]
    network_style_encoder_script: Optional[torch.nn.Module]


@dataclass
class BasePos:
    root_pos: Tensor
    root_rot: Tensor
    root_vel: Tensor
    root_vrt: Tensor
    lpos: Tensor
    ltxy: Tensor
    lvel: Tensor
    lvrt: Tensor
    gaze_pos: Tensor


class GestureInferenceModel:
    def __init__(self, options_path: str = "./data/zeggs/options.json", style: str = "Happy"):
        with open(options_path, "r") as f:
            options = json.load(f)

        self.train_options = options["train_opt"]
        self.network_options = options["net_opt"]
        paths = options["paths"]

        self.base_path = Path(paths["base_path"])
        self.data_path = self.base_path / paths["path_processed_data"]

        self.network_path = Path(paths["models_dir"])
        self.output_path = Path(paths["output_dir"])

        self.results_path = Path(self.output_path) / "results"

        self.style_encoding_type = "label"
        self.style = [style]

        # self.style_encoding_type = "example"
        # self.style = [(Path(style_path), None)]

        self.blend_type = "stitch"
        self.blend_ratio = [0.5, 0.5]
        self.config: Optional[GestureInferenceModelConfig] = None
        self.style_encoding: List = []
        self.base_pos: Optional[BasePos] = None
        self.prev_anim: Optional[dict] = None

    def load_style_encoding(self, temperature: float = 1.0) -> Tuple[List[Any], BasePos]:
        config = self.config

        assert config is not None

        device = config.device

        with torch.no_grad():
            style_encodings = []

            for style in self.style:
                if self.style_encoding_type == "example":
                    anim_data = bvh.load(style[0])

                    anim_fps = int(np.ceil(1 / anim_data["frametime"]))
                    assert anim_fps == 60

                    # Extracting features
                    (
                        root_pos,
                        root_rot,
                        root_vel,
                        root_vrt,
                        lpos,
                        lrot,
                        ltxy,
                        lvel,
                        lvrt,
                        cpos,
                        crot,
                        ctxy,
                        cvel,
                        cvrt,
                        gaze_pos,
                        gaze_dir,
                    ) = preprocess_animation(anim_data)

                    # convert to tensor
                    nframes = len(anim_data["rotations"])
                    root_vel = torch.as_tensor(
                        root_vel, dtype=torch.float32, device=device)
                    root_vrt = torch.as_tensor(
                        root_vrt, dtype=torch.float32, device=device)
                    root_pos = torch.as_tensor(
                        root_pos, dtype=torch.float32, device=device)
                    root_rot = torch.as_tensor(
                        root_rot, dtype=torch.float32, device=device)
                    lpos = torch.as_tensor(
                        lpos, dtype=torch.float32, device=device)
                    ltxy = torch.as_tensor(
                        ltxy, dtype=torch.float32, device=device)
                    lvel = torch.as_tensor(
                        lvel, dtype=torch.float32, device=device)
                    lvrt = torch.as_tensor(
                        lvrt, dtype=torch.float32, device=device)
                    gaze_pos = torch.as_tensor(
                        gaze_pos, dtype=torch.float32, device=device)

                    base_pos = BasePos(
                        root_pos=root_pos,
                        root_rot=root_rot,
                        root_vel=root_vel,
                        root_vrt=root_vrt,
                        lpos=lpos,
                        ltxy=ltxy,
                        lvel=lvel,
                        lvrt=lvrt,
                        gaze_pos=gaze_pos,
                    )

                    S_root_vel = root_vel.reshape(nframes, -1)
                    S_root_vrt = root_vrt.reshape(nframes, -1)
                    S_lpos = lpos.reshape(nframes, -1)
                    S_ltxy = ltxy.reshape(nframes, -1)
                    S_lvel = lvel.reshape(nframes, -1)
                    S_lvrt = lvrt.reshape(nframes, -1)
                    example_feature_vec = torch.cat(
                        [
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            torch.zeros_like(S_root_vel),
                        ],
                        dim=1,
                    )
                    example_feature_vec = (
                        example_feature_vec - config.anim_input_mean) / config.anim_input_std

                    assert config.network_style_encoder_script is not None

                    style_encoding, _, _ = config.network_style_encoder_script(
                        example_feature_vec[np.newaxis], temperature
                    )
                    style_encodings.append(style_encoding)
                elif self.style_encoding_type == "label":
                    style_index = config.labels.index(style)
                    # style_index = style
                    style_embeddding = torch.zeros((1, 64), dtype=torch.float32, device=device)
                    style_embeddding[0, style_index] = 1.0
                    style_encodings.append(style_embeddding)

                    base_pos = self.load_first_pose(self.prev_anim)
                else:
                    raise ValueError("Unknown style encoding type")

        return (style_encodings, base_pos)

    def load_config(self, seed: int = 1234, use_gpu: bool = False) -> GestureInferenceModelConfig:
        path_network_speech_encoder_weights = self.network_path / "speech_encoder.pt"
        path_network_decoder_weights = self.network_path / "decoder.pt"
        if self.style_encoding_type == "example":
            path_network_style_encoder_weights = self.network_path / "style_encoder.pt"
        path_stat_data = self.data_path / "stats.npz"
        path_data_definition = self.data_path / "data_definition.json"
        path_data_pipeline_conf = self.data_path / "data_pipeline_conf.json"
        if self.results_path is not None:
            self.results_path.mkdir(exist_ok=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(10)
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Data pipeline conf (We must use the same processing configuration as the one in training)
        with open(path_data_pipeline_conf, "r") as f:
            data_pipeline_conf = json.load(f)
        data_pipeline_conf = DictConfig(data_pipeline_conf)

        # Animation static info (Skeleton, FPS, etc)
        with open(path_data_definition, "r") as f:
            details = json.load(f)

        label_names = details["label_names"]
        bone_names = details["bone_names"]
        parents = torch.as_tensor(
            details["parents"], dtype=torch.long, device=device)
        dt = details["dt"]

        # Load Stats (Mean and Std of input/output)

        stat_data = np.load(path_stat_data)
        audio_input_mean = torch.as_tensor(
            stat_data["audio_input_mean"], dtype=torch.float32, device=device
        )
        audio_input_std = torch.as_tensor(
            stat_data["audio_input_std"], dtype=torch.float32, device=device
        )
        anim_input_mean = torch.as_tensor(
            stat_data["anim_input_mean"], dtype=torch.float32, device=device
        )
        anim_input_std = torch.as_tensor(
            stat_data["anim_input_std"], dtype=torch.float32, device=device
        )
        anim_output_mean = torch.as_tensor(
            stat_data["anim_output_mean"], dtype=torch.float32, device=device
        )
        anim_output_std = torch.as_tensor(
            stat_data["anim_output_std"], dtype=torch.float32, device=device
        )

        # Load Networks
        network_speech_encoder = torch.load(
            path_network_speech_encoder_weights, map_location=device).to(device)
        network_speech_encoder.eval()

        network_decoder = torch.load(
            path_network_decoder_weights, map_location=device).to(device)
        network_decoder.eval()

        network_style_encoder = None
        network_style_encoder_script = None

        if self.style_encoding_type == "example":
            network_style_encoder = torch.load(
                path_network_style_encoder_weights, map_location=device).to(device)
            network_style_encoder.eval()

        network_speech_encoder_script = network_speech_encoder
        network_decoder_script = network_decoder
        if self.style_encoding_type == "example":
            network_style_encoder_script = network_style_encoder

        network_speech_encoder_script.eval()
        network_decoder_script.eval()
        if self.style_encoding_type == "example":
            network_style_encoder_script.eval()

        return GestureInferenceModelConfig(
            device=device,
            data_pipeline_conf=data_pipeline_conf,
            labels=label_names,
            bones=bone_names,
            parents=parents,
            dt=dt,
            audio_input_mean=audio_input_mean,
            audio_input_std=audio_input_std,
            anim_input_mean=anim_input_mean,
            anim_input_std=anim_input_std,
            anim_output_mean=anim_output_mean,
            anim_output_std=anim_output_std,
            network_speech_encoder=network_speech_encoder,
            network_decoder=network_decoder,
            network_style_encoder=network_style_encoder,
            network_speech_encoder_script=network_speech_encoder_script,
            network_decoder_script=network_decoder_script,
            network_style_encoder_script=network_style_encoder_script,
        )

    @timeit
    def load_model(self):
        # Load config
        self.config = self.load_config(use_gpu=True)

    def load_first_pose(self, anim_data: Optional[dict[str, Any]] = None) -> BasePos:
        device = self.config.device

        if anim_data is None:
            current_dir = Path(__file__).parent
            anim_data = bvh.load(f"{current_dir}/../../data/zeggs/styles/first_pose.bvh")

        (
            root_pos,
            root_rot,
            root_vel,
            root_vrt,
            lpos,
            lrot,
            ltxy,
            lvel,
            lvrt,
            cpos,
            crot,
            ctxy,
            cvel,
            cvrt,
            gaze_pos,
            gaze_dir,
        ) = preprocess_animation(anim_data)

        root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=device)
        root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=device)
        root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
        root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=device)
        lpos = torch.as_tensor(lpos, dtype=torch.float32, device=device)
        ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=device)
        lvel = torch.as_tensor(lvel, dtype=torch.float32, device=device)
        lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=device)
        gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=device)

        return BasePos(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_vrt=root_vrt,
            lpos=lpos,
            ltxy=ltxy,
            lvel=lvel,
            lvrt=lvrt,
            gaze_pos=gaze_pos,
        )

    @timeit
    def generate(self, audio_file_path: Optional[str]) -> Optional[str]:
        config = self.config

        assert config is not None, "Model not loaded"

        self.style_encoding, self.base_pos = self.load_style_encoding()

        with torch.no_grad():
            if audio_file_path is None:
                audio_features = torch.zeros(
                    (242, 81), device=config.device, dtype=torch.float32
                )
            else:
                # Load Audio
                _, audio_data = read_wavfile(
                    audio_file_path,
                    rescale=True,
                    desired_fs=16000,
                    desired_nb_channels=None,
                    out_type="float32",
                    logger=None,
                )

                n_frames = int(round(60.0 * (len(audio_data) / 16000)))

                audio_features = torch.as_tensor(
                    preprocess_audio(
                        audio_data,
                        60,
                        n_frames,
                        config.data_pipeline_conf.audio_conf,
                        feature_type=config.data_pipeline_conf.audio_feature_type,
                    ),
                    device=config.device,
                    dtype=torch.float32,
                )

                audio_features = torch.nan_to_num(audio_features)

            speech_encoding = config.network_speech_encoder_script(
                (audio_features[np.newaxis] -
                 config.audio_input_mean) / config.audio_input_std
            )
            final_style_encoding: Any = None

            if self.blend_type == "stitch":
                if len(self.style_encoding) > 1:
                    assert len(self.style) == len(self.blend_ratio)
                    se = split_by_ratio(n_frames, self.blend_ratio)
                    V_root_pos = []
                    V_root_rot = []
                    V_lpos = []
                    V_ltxy = []
                    final_style_encoding = []
                    for i, style_encoding in enumerate(self.style_encoding):
                        final_style_encoding.append(
                            style_encoding.unsqueeze(1).repeat(
                                (1, se[i][-1] - se[i][0], 1))
                        )
                    final_style_encoding = torch.cat(
                        final_style_encoding, dim=1)
                else:
                    final_style_encoding = self.style_encoding[0]
            elif self.blend_type == "add":
                # style_encoding = torch.mean(torch.stack(style_encodings), dim=0)
                if len(self.style_encoding) > 1:
                    assert len(self.style_encoding) == len(self.blend_ratio)
                    final_style_encoding = torch.matmul(
                        torch.stack(self.style_encoding,
                                    dim=1).transpose(2, 1),
                        torch.tensor(self.blend_ratio, device=config.device),
                    )
                else:
                    final_style_encoding = self.style_encoding[0]

            base_pos = self.base_pos

            assert base_pos is not None

            root_pos_0 = base_pos.root_pos[0][np.newaxis]
            root_rot_0 = base_pos.root_rot[0][np.newaxis]
            root_vel_0 = base_pos.root_vel[0][np.newaxis]
            root_vrt_0 = base_pos.root_vrt[0][np.newaxis]
            lpos_0 = base_pos.lpos[0][np.newaxis]
            ltxy_0 = base_pos.ltxy[0][np.newaxis]
            lvel_0 = base_pos.lvel[0][np.newaxis]
            lvrt_0 = base_pos.lvrt[0][np.newaxis]

            if final_style_encoding.dim() == 2:
                final_style_encoding = final_style_encoding.unsqueeze(
                    1).repeat((1, speech_encoding.shape[1], 1))
            (
                V_root_pos,
                V_root_rot,
                V_root_vel,
                V_root_vrt,
                V_lpos,
                V_ltxy,
                V_lvel,
                V_lvrt,
            ) = config.network_decoder_script(
                root_pos_0,
                root_rot_0,
                root_vel_0,
                root_vrt_0,
                lpos_0,
                ltxy_0,
                lvel_0,
                lvrt_0,
                base_pos.gaze_pos[0: 0 + 1].repeat_interleave(speech_encoding.shape[1], dim=0)[
                    np.newaxis
                ],
                speech_encoding,
                final_style_encoding,
                config.parents,
                config.anim_input_mean,
                config.anim_input_std,
                config.anim_output_mean,
                config.anim_output_std,
                config.dt,
            )

            V_lrot = quat.from_xform(
                xform_orthogonalize_from_xy(V_ltxy).detach().cpu().numpy())

            result_path = os.path.join(tempfile.gettempdir(), f"{uuid4()}.bvh")

            try:
                bvh_data = write_bvh(
                    result_path,
                    V_root_pos[0].detach().cpu().numpy(),
                    V_root_rot[0].detach().cpu().numpy(),
                    V_lpos[0].detach().cpu().numpy(),
                    V_lrot[0],
                    parents=config.parents.detach().cpu().numpy(),
                    names=config.bones,
                    order="zyx",
                    dt=config.dt,
                    start_position=np.array([0, 0, 0]),
                    start_rotation=np.array([1, 0, 0, 0]),
                )

                self.prev_anim = bvh_data

                # take last 4 frames only
                if self.prev_anim["rotations"].shape[0] > 4:
                    self.prev_anim["rotations"] = self.prev_anim["rotations"][-4:]
                    self.prev_anim["positions"] = self.prev_anim["positions"][-4:]

                return result_path
            except (PermissionError, OSError) as e:
                print(e)
                return None

    @timeit
    def post_process(self, result_path: str, dest_path: Optional[str] = None) -> str:
        fixed_path = dest_path or os.path.join(tempfile.gettempdir(), f"{uuid4()}.bvh")

        reset_pose(result_path, fixed_path)

        with open(fixed_path, "r") as f:
            result = f.read()

        os.remove(result_path)

        if dest_path is None:
            os.remove(fixed_path)

        return result

    @timeit
    def infer(self, audio_file_path: Optional[str], dest_path: Optional[str] = None) -> Optional[str]:
        result_path = self.generate(audio_file_path)
        if result_path is None:
            return None

        return self.post_process(result_path, dest_path)

    @timeit
    def infer_motions(self, audio_file_path: Optional[str]) -> List[str]:
        result_path = self.generate(audio_file_path)
        if result_path is None:
            return []

        with open(result_path, "r") as f:
            result = f.readlines()
            return result[464:-2]


if __name__ == "__main__":
    model = GestureInferenceModel()

    model.load_model()
    model.infer("data/samples/barefoot.wav", "part1.bvh")
    model.infer("data/samples/barefoot.wav", "part2.bvh")
