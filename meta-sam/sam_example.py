import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from segment_anything import SamPredictor, sam_model_registry

def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_mask(mask, ax, random_color=False):
    """Display a mask on the given axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Display points on the given axis."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Display a bounding box on the given axis."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))

class Timer:
    """A simple timer class to measure execution time."""
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()

    def elapsed_ms(self):
        """Return the elapsed time in milliseconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer has not been started or stopped.")
        return (self.end_time - self.start_time) * 1000
    
class InteractiveSegmentation:
    def __init__(self, image_path, model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth"):
         # Initialize a timer for measuring performance
        self.timer = Timer()

        # Load the SAM model
        self.device = "cpu"
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

        # Load and encode the image
        self.image = load_image(image_path)
        print(f"Encoding the image with device={self.device}...")
        self.timer.start()
        self.predictor.set_image(self.image)
        self.timer.stop()
        print(f"Image encoding completed in {self.timer.elapsed_ms():.2f} ms.\n")
       

    def get_mask_from_prompt(self, prompt_type, prompt_data):
        """Generate a segmentation mask based on the given prompt."""
        if prompt_type == "point":
            input_point = np.array(prompt_data["points"])
            input_label = np.array(prompt_data["labels"])
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
        elif prompt_type == "box":
            input_box = np.array(prompt_data["box"])
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True
            )
        else:
            raise ValueError("Unsupported prompt type. Choose 'point' or 'box'.")
        
        return masks, scores

    def visualize_masks(self, masks, scores, prompt_type=None, prompt_data=None):
        """Visualize the segmentation masks."""
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            show_mask(mask, plt.gca())
            if prompt_type == "point":
                show_points(np.array(prompt_data["points"]), np.array(prompt_data["labels"]), plt.gca())
            elif prompt_type == "box":
                show_box(prompt_data["box"], plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

def main():
    # User inputs
    image_path = input("Enter the path to the image: ")
    segmentation_tool = InteractiveSegmentation(image_path)

    while True:
        print("\nChoose a prompt type:")
        print("1. Point")
        print("2. Box")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":  # Point prompt
            points = eval(input("Enter the points as a list of [x, y] coordinates: "))
            labels = eval(input("Enter the labels (1 for foreground, 0 for background) as a list: "))
            prompt_data = {"points": points, "labels": labels}
            masks, scores = segmentation_tool.get_mask_from_prompt("point", prompt_data)
            segmentation_tool.visualize_masks(masks, scores, prompt_type="point", prompt_data=prompt_data)

        elif choice == "2":  # Box prompt
            box = eval(input("Enter the bounding box as [x_min, y_min, x_max, y_max]: "))
            prompt_data = {"box": box}
            masks, scores = segmentation_tool.get_mask_from_prompt("box", prompt_data)
            segmentation_tool.visualize_masks(masks, scores, prompt_type="box", prompt_data=prompt_data)

        elif choice == "3":  # Exit
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()