
import time
from typing import Union

import numpy as np
import torch
from PIL import Image

from src.gradcam import (
    generate_gradcam,
    setup_gradcam,
)

from src.transforms import (
    get_validation_transforms,
)


def load_image(
    image: Union[str, Image.Image, np.ndarray],
) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def preprocess_image(
    image: Image.Image,
) -> torch.Tensor:
    transform = get_validation_transforms()
    return transform(image)


def predict_image(
    image,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list,
    return_gradcam: bool = True,
) -> dict:
    start_time = time.perf_counter()

    image = load_image(
        image
    )

    image_tensor = preprocess_image(
        image
    )

    input_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        ).squeeze()

    prediction = probabilities.argmax().item()

    confidence = probabilities[
        prediction
    ].item()

    probability_dict = {

        class_names[index]: float(probabilities[index])

        for index in range(
            len(class_names)
        )

    }

    gradcam_image = None

    if return_gradcam:

        cam = setup_gradcam(
            model
        )

        gradcam_image = generate_gradcam(

            cam,

            image_tensor,

            prediction,

        )

    inference_time = (

        time.perf_counter()

        - start_time

    ) * 1000

    return {

        "predicted_class": class_names[
            prediction
        ],

        "confidence": confidence,

        "probabilities": probability_dict,

        "gradcam_image": gradcam_image,

        "inference_time_ms": inference_time,

    }


def predict_from_path(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list,
):

    return predict_image(

        image=image_path,

        model=model,

        device=device,

        class_names=class_names,

    )


def predict_from_numpy(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    class_names: list,
):


    return predict_image(

        image=image,

        model=model,

        device=device,

        class_names=class_names,

    )


def print_prediction(
    prediction: dict,
) -> None:
    """
    Pretty-print prediction.
    """

    print()

    print("=" * 60)

    print("Prediction")

    print("=" * 60)

    print(

        f"Class       : "

        f"{prediction['predicted_class']}"

    )

    print(

        f"Confidence  : "

        f"{prediction['confidence']*100:.2f}%"

    )

    print(

        f"Inference   : "

        f"{prediction['inference_time_ms']:.2f} ms"

    )

    print()

    print("Probabilities")

    print("-" * 60)

    for label, probability in prediction[
        "probabilities"
    ].items():

        print(

            f"{label:12s}: "

            f"{probability:.4f}"

        )

    print("=" * 60)