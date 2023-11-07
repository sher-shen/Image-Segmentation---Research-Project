import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from loadingpy import pybar
from torchinfo import summary
import time

from .architecture import Segmentor
from .dataloader import create_dataloader
from .loss import BalancedLoss


class BlankStatement:
    def __init__(self):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        return None


class EndNestedLoop(Exception):
    def __init__(self, message="", errors=""):
        super().__init__(message)
        self.errors = errors


class MPScaler:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *args: Any,
        **kwds: Any,
    ) -> Any:
        if self.scaler is None:
            loss.backward()
            optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

class BaselineTrainer:
    def __init__(self, quiet_mode: bool) -> None:
        self.quiet_mode = quiet_mode
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        if not os.path.exists(os.path.join(os.getcwd(), "submissions")):
            os.mkdir(os.path.join(os.getcwd(), "submissions"))
        if not os.path.exists(os.path.join(os.getcwd(), "submissions", "baseline")):
            os.mkdir(os.path.join(os.getcwd(), "submissions", "baseline"))
        self.sigmoid = torch.nn.Sigmoid()
        self.scope = (
            torch.cuda.amp.autocast() if torch.cuda.is_available() else BlankStatement()
        )
        self.train_losses = []
        self.losses = []
        self.score = []
        self.single_student_folder_path = os.path.join(os.getcwd(), "single_student_predictions")

        if not os.path.exists(self.single_student_folder_path):
            os.makedirs(self.single_student_folder_path)

        self.labels_folder = os.path.join(os.getcwd(), "student_set", "val")
        self.label_paths = [elem for elem in sorted(os.listdir(self.labels_folder)) if "label" in elem]

    def make_final_predictions(
        self, model: torch.nn.Module, batch_size: int = 16
    ) -> None:
        dataset = create_dataloader(
            path_to_data=os.path.join(os.getcwd(), "student_set", "test"),
            batch_size=batch_size,
        )
        model.eval()
        cpt = 0
        with torch.no_grad():
            with self.scope:
                for inputs, original_sizes in pybar(
                    dataset, base_str="extract on the test set"
                ):
                    inputs = inputs.to(self.device)
                    predictions = torch.round(model(inputs))
                    for prediction, original_size in zip(predictions, original_sizes):
                        _, W, H = original_size
                        prediction = torchvision.transforms.Resize(
                            size=(W, H), antialias=True
                        )(prediction)
                        np.save(
                            os.path.join(
                                os.getcwd(),
                                "submissions",
                                "baseline",
                                str(cpt).zfill(6) + ".npy",
                            ),
                            torch.round(self.sigmoid(prediction)).cpu().numpy(),
                        )
                        cpt += 1

    def compute_iou(self, prediction: torch.Tensor, label: torch.Tensor) -> float:
        # Convert tensors to numpy arrays
        p = prediction.cpu().numpy().astype(float)
        g = label.cpu().numpy().astype(float)

        i = np.sum(p * g)   # Intersection
        u = np.sum(p) + np.sum(g) - i  # Union
        if u == 0:
            return -1
        return i / u

    def evaluate_one_student(self, student_folder: str) -> float:
        scores = []
        prediction_paths = [elem for elem in sorted(os.listdir(student_folder))]
        for pred_path, label_path in zip(prediction_paths, self.label_paths):
            score = self.compute_iou(
                pred=np.load(os.path.join(student_folder, pred_path)),
                label=np.load(os.path.join(self.labels_folder, label_path)),
            )
            if score < 0:
                print(label_path)
            scores.append(score)
        return np.mean(scores)

    def calculate_val_metrics(self, model, dataset, loss_fn) -> tuple:
        model.eval()
        total_loss = 0
        num_batches = 0
        scores = []

        with torch.no_grad():
            for inputs, labels in dataset:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                predictions = model(inputs)
                loss = loss_fn(predictions, labels)
                total_loss += loss.item()

                # Convert predictions to binary values (0 or 1) using a threshold of 0.5
                bin_predictions = torch.where(predictions > 0.5, torch.tensor(1.0).to(self.device),
                                              torch.tensor(0.0).to(self.device))
                for prediction, label in zip(bin_predictions, labels):
                    score = self.compute_iou(prediction, label)
                    if score >= 0:  # Check to ensure the union isn't zero
                        scores.append(score)
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_score = np.mean(scores) if scores else 0  # This replaces the IoU computation

        return avg_loss, avg_score
    

    def train(self, num_opt_steps: int = 20000, batch_size: int = 16, lr: float = 5.0e-3) -> None:
        start_time = time.time()
        print( batch_size)
        dataset = create_dataloader(
            path_to_data=os.path.join(os.getcwd(), "student_set", "train"),
            batch_size=batch_size,
        )
        model = Segmentor()
        if not self.quiet_mode:
            summary(model, input_size=(batch_size, 3, 224, 224))
        model = model.to(self.device)
        scaler = MPScaler()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        loss_fn = BalancedLoss()

        total_data_size = len(dataset.dataset)
        num_batches_per_epoch = total_data_size // batch_size
        num_epochs = num_opt_steps // num_batches_per_epoch
        print(num_opt_steps)
        print(total_data_size)
        print(batch_size)
        patience = 5
        #best_val_loss = float('inf')
        best_val_score = float('-inf')
        no_improve_epochs = 0
        val_dataset = create_dataloader(
            path_to_data=os.path.join(os.getcwd(), "student_set", "val"),
            batch_size=batch_size,
        )
        model_save_path = os.path.join(os.getcwd(), "src", "best_model", "best_model.pth")

        # Ensure the directory for model_save_path exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        for epoch in range(num_epochs):
            pbar = pybar(dataset, base_str=f"Epoch {epoch+1}/{num_epochs} - training")
            train_losses_for_epoch = []
            for inputs, labels in pbar:
                optimizer.zero_grad()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                predictions = model(inputs)
                loss = loss_fn(predictions, labels)
                scaler(loss=loss, optimizer=optimizer)
                train_losses_for_epoch.append(loss.item())

            avg_train_loss = sum(train_losses_for_epoch) / len(train_losses_for_epoch)
            self.train_losses.append(avg_train_loss)

            avg_loss, avg_score = self.calculate_val_metrics(model, val_dataset, loss_fn)
            self.losses.append(avg_loss)
            self.score.append(avg_score.item() if torch.is_tensor(avg_score) else avg_score)

            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_loss:.4f}, Score: {avg_score:.4f}")
            if avg_score > best_val_score:
                best_val_score = avg_score
                no_improve_epochs = 0
                # Save the model state when a new best is found
                torch.save(model.state_dict(), model_save_path)
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break  # This breaks out of the training loop
            if not self.quiet_mode:
                pbar.set_description(
                    description=f"loss: {loss.cpu().detach().numpy():.4f}"
                )
        end_time = time.time()
        total_time = end_time - start_time 
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Training completed in: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        save_path = os.path.join(os.getcwd(), "plots")
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Prepare the plot
        fig, ax1 = plt.subplots()

        # Set labels for x and y axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Metrics')

        # Plotting Train and Val loss on ax1
        ax1.plot(self.train_losses, label='Train Loss', color='tab:blue')
        ax1.plot(self.losses, label='Val Loss', color='tab:orange')

        # Plotting IoU on ax1
        ax1.plot(self.score, label='Val score', color='tab:green', linestyle='--')

        # Legend to differentiate the plots
        ax1.legend(loc='upper left')

        # # Save the plot to a file
        plot_filename = os.path.join(save_path, "training_validation_loss_score.png")
        plt.savefig(plot_filename)
        plt.close()
        try:
            pbar.__next__()
        except StopIteration:
            pass
        model.load_state_dict(torch.load(model_save_path))
        self.make_final_predictions(model=model, batch_size=batch_size)
