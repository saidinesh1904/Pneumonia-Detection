

from configs.config import (
    DATA_DIR,
    FINAL_MODEL_PATH,
    IMG_SIZE,
)

from src.utils import (
    create_directories,
    get_device,
    print_environment,
    set_seed,
)

from src.download_dataset import (
    download_dataset,
)

from src.dataset import (
    create_datasets,
    create_dataloaders,
)

from src.model import (
    build_model,
    move_model_to_device,
    print_model_summary,
)

from src.losses import (
    initialize_training,
)

from src.trainer import (
    train_model,
)

from src.evaluator import (
    evaluate_model,
)

from src.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    visualize_predictions,
)

from src.gradcam import (
    visualize_gradcam,
)

from src.benchmark import (
    benchmark_inference,
)

from src.checkpoint import (
    save_checkpoint,
)


def main():

    # ======================================================
    # Environment
    # ======================================================

    create_directories()

    set_seed()

    print_environment()

    device = get_device()

    # ======================================================
    # Dataset
    # ======================================================

    download_dataset()

    train_dataset, val_dataset, test_dataset = create_datasets(
        DATA_DIR
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        device,
    )

    class_names = train_dataset.classes

    # ======================================================
    # Model
    # ======================================================

    model = build_model()

    model = move_model_to_device(
        model,
        device,
    )

    print_model_summary(
        model
    )

    # ======================================================
    # Training Components
    # ======================================================

    (
        criterion,
        optimizer,
        scheduler,
        scaler,
    ) = initialize_training(

        model,

        train_dataset,

        device,

    )

    # ======================================================
    # Train
    # ======================================================

    model, history = train_model(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        scaler=scaler,

        device=device,

        class_names=class_names,

    )

    # ======================================================
    # Evaluation
    # ======================================================

    results = evaluate_model(

        model,

        test_loader,

        device,

        class_names,

    )

    # ======================================================
    # Visualizations
    # ======================================================

    plot_training_curves(
        history
    )

    plot_confusion_matrix(

        results["labels"],

        results["predictions"],

        class_names,

    )

    plot_roc_pr_curves(

        results["labels"],

        results["probabilities"],

    )

    visualize_predictions(

        model,

        test_dataset,

        device,

        class_names,

    )

    visualize_gradcam(

        model,

        test_dataset,

        device,

        class_names,

    )

    # ======================================================
    # Benchmark
    # ======================================================

    benchmark_inference(

        model,

        device,

    )

    # ======================================================
    # Save Final Model
    # ======================================================

    save_checkpoint(

        model=model,

        optimizer=optimizer,

        epoch=len(history["train_loss"]),

        val_accuracy=max(history["val_acc"]),

        class_names=class_names,

        image_size=IMG_SIZE,

        filepath=FINAL_MODEL_PATH,

    )

    print()

    print("=" * 70)

    print("Pipeline Completed Successfully")

    print("=" * 70)


if __name__ == "__main__":

    main()