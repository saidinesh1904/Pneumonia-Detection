"""
Gradio application for Pneumonia Detection.
"""

import gradio as gr
import torch

from configs.config import FINAL_MODEL_PATH
from src.checkpoint import load_model
from src.inference import predict_image
from src.utils import get_device



device = get_device()

model, checkpoint = load_model(
    FINAL_MODEL_PATH,
    device,
)

CLASS_NAMES = checkpoint["class_names"]



def gradio_predict(image):

    if image is None:

        return {}, None, "Please upload a chest X-ray."

    result = predict_image(
        image=image,
        model=model,
        device=device,
        class_names=CLASS_NAMES,
        return_gradcam=True,
    )

    probabilities = {
        label: float(score)
        for label, score in result["probabilities"].items()
    }

    information = (
        f"Prediction : {result['predicted_class']}\n\n"
        f"Confidence : {result['confidence']*100:.2f}%\n\n"
        f"Inference Time : {result['inference_time_ms']:.2f} ms"
    )

    return (
        probabilities,
        result["gradcam_image"],
        information,
    )



with gr.Blocks(
    title="Pneumonia Detection",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
# 🫁 Pneumonia Detection using EfficientNet-B0

Upload a chest X-ray image to classify it as:

- NORMAL
- PNEUMONIA

The application also provides a Grad-CAM heatmap showing
which regions influenced the model's prediction.
"""
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                label="Chest X-Ray",
                type="numpy",
                height=320,
            )

            predict_button = gr.Button(
                "Analyze Image",
                variant="primary",
            )

        with gr.Column():

            prediction_output = gr.Label(
                label="Class Probabilities",
            )

            info_output = gr.Textbox(
                label="Prediction Details",
                lines=6,
            )

    gradcam_output = gr.Image(
        label="Grad-CAM",
        height=350,
    )

    predict_button.click(
        fn=gradio_predict,
        inputs=input_image,
        outputs=[
            prediction_output,
            gradcam_output,
            info_output,
        ],
    )

    gr.Markdown(
        """
---
### Grad-CAM Interpretation

- 🔴 Bright regions indicate areas receiving the highest model attention.
- 🔵 Cooler regions indicate lower importance.

This visualization helps explain why the model predicted a particular class.
"""
    )


if __name__ == "__main__":

    demo.launch(
        share=False,
        debug=False,
    )