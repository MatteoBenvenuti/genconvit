import math
import os
import cv2
import pandas as pd
import torch
from model.config import load_config
from model.pred_func import (
    df_face,
    is_video,
    real_or_fake,
    set_result,
    store_result,
)
import torch.nn as nn
from model.genconvit_ed import GenConViTED
from model.genconvit_vae import GenConViTVAE

config = load_config()
print("CONFIG")
print(config)


class CustomGenConViT(nn.Module):
    def __init__(self, config, ed, vae, net, fp16):
        super(CustomGenConViT, self).__init__()
        self.net = net
        self.fp16 = fp16
        if self.net == "ed":
            try:
                self.model_ed = GenConViTED(config)
                self.checkpoint_ed = torch.load(
                    f"weight/{ed}.pth", map_location=torch.device("cpu")
                )

                if "state_dict" in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed["state_dict"])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)

                self.model_ed.eval()
                if self.fp16:
                    self.model_ed.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
        elif self.net == "vae":
            try:
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_vae = torch.load(
                    f"weight/{vae}.pth", map_location=torch.device("cpu")
                )

                if "state_dict" in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae["state_dict"])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)

                self.model_vae.eval()
                if self.fp16:
                    self.model_vae.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{vae}.pth file not found.")
        else:
            try:
                self.model_ed = GenConViTED(config)
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_ed = torch.load(
                    f"weight/{ed}.pth", map_location=torch.device("cpu")
                )
                self.checkpoint_vae = torch.load(
                    f"weight/{vae}.pth", map_location=torch.device("cpu")
                )
                if "state_dict" in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed["state_dict"])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)
                if "state_dict" in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae["state_dict"])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                self.model_ed.eval()
                self.model_vae.eval()
                if self.fp16:
                    self.model_ed.half()
                    self.model_vae.half()
            except FileNotFoundError:
                raise Exception("Error: Model weights file not found.")

    def forward(self, x):
        x1 = self.model_ed(x)
        x2, _ = self.model_vae(x)
        x = torch.cat((x1, x2), dim=0)  # (x1+x2)/2 #
        return x1, x2, x


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_genconvit(config, net, ed_weight, vae_weight, fp16):
    model = CustomGenConViT(config, ed=ed_weight, vae=vae_weight, net=net, fp16=fp16)

    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


def compute_num_frames(video_path, fps):
    # Apri il video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")

    # Ottieni FPS originali e numero di frame
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcola la durata in secondi
    duration = total_frames / original_fps if original_fps > 0 else 0

    # Calcola quanti frame estrarre in base al nuovo fps specificato
    frame_to_extract = math.floor(duration * fps)

    cap.release()
    return duration, frame_to_extract


def pred_vid(df, model, batch_size=8):
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    
    all_x1 = []
    all_x2 = []
    all_x = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]
            x1, x2, x = model(batch)

            all_x1.append(sigmoid(x1.squeeze()))
            all_x2.append(sigmoid(x2.squeeze()))
            all_x.append(sigmoid(x.squeeze()))

    # Concatenazione risultati su batch
    all_x1 = torch.cat(all_x1, dim=0)
    all_x2 = torch.cat(all_x2, dim=0)
    all_x = torch.cat(all_x, dim=0)

    return all_x, all_x1, all_x2, max_prediction_value(all_x)


def max_prediction_value(y_pred):
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def predict(
    vid,
    model,
    fp16,
    result,
    num_frames,
    net,
    klass,
    count=0,
    accuracy=-1,
    correct_label="unknown",
    compression=None,
):
    count += 1
    print(f"\n\n{str(count)} Loading... {vid}")

    df = df_face(vid, num_frames, net)  # extract face from the frames

    if len(df) == 0:
        raise Exception("volti non riconosciuti")

    if fp16:
        df.half()
    x1, x2, x, (y, y_val) = (
        pred_vid(df, model)
        if len(df) >= 1
        else (torch.tensor(0).item(), torch.tensor(0.5).item())
    )
    result = store_result(
        result, os.path.basename(vid), y, y_val, klass, correct_label, compression
    )

    if accuracy > -1:
        if correct_label == real_or_fake(y):
            accuracy += 1
        print(
            f"\nPrediction: {y_val} {real_or_fake(y)} \t\t {accuracy}/{count} {accuracy / count}"
        )

    return result, accuracy, count, [y, y_val], (x, x1, x2)


def vids(
    output_file_name,
    ed_weight,
    vae_weight,
    root_dir,
    fps,
    net=None,
    fp16=False,
):
    result = set_result()
    r = 0
    f = 0
    count = 0

    custom_result_path = os.path.join("./custom_results", output_file_name)

    result_custom = []

    model = load_genconvit(config, net, ed_weight, vae_weight, fp16)

    for filename in os.listdir(root_dir):
        curr_vid = os.path.join(root_dir, filename)

        try:
            if filename == ".gitignore":
                continue

            if is_video(curr_vid):
                video_length, num_frames = compute_num_frames(curr_vid, fps)
                print(f"{video_length=},{num_frames=}")
                result, accuracy, count, pred, (x, x1, x2) = predict(
                    curr_vid,
                    model,
                    fp16,
                    result,
                    num_frames,
                    net,
                    "uncategorized",
                    count,
                )
                f, r = (f + 1, r) if "FAKE" == real_or_fake(pred[0]) else (f, r + 1)
                print(
                    f"Prediction: {pred[1]} {real_or_fake(pred[0])} \t\tFake: {f} Real: {r}"
                )

                result_custom.append(
                    {
                        "filename": filename,
                        "video_length": video_length,
                        "frames": num_frames,
                        "genconvit": x.tolist(),
                        "ed": x1.tolist(),
                        "vae": x2.tolist(),
                    }
                )
                pd.DataFrame(result_custom).to_csv(custom_result_path, index=False)
            else:
                print(
                    f"Invalid video file: {curr_vid}. Please provide a valid video file."
                )

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    pd.DataFrame(result_custom).to_csv(custom_result_path, index=False)

    return result


if __name__ == "__main__":
    net = "genconvit"
    ed_weight = "genconvit_ed_inference"
    vae_weight = "genconvit_vae_inference"

    output_file_name = "result.csv"
    root_dir = "./data"
    fps = 3

    vids(
        output_file_name,
        ed_weight,
        vae_weight,
        root_dir,
        fps,
        net,
        False,
    )
