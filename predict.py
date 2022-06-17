from pathlib import Path
import pathlib
import os

import torch
import torchio as tio
import fire

from segmentation_pipeline.utils.torch_context import TorchContext
from segmentation_pipeline.post_processing import remove_holes, keep_components
from segmentation_pipeline.models.ensemble import EnsembleFlips, EnsembleModels
from segmentation_pipeline.transforms.replace_nan import ReplaceNan
from segmentation_pipeline.data_processing.subject_loaders import ImageLoader


# Hack so that models can load on windows
if os.name == 'nt':
    pathlib.PosixPath = pathlib.Path


def inference(subjects, predictor, model, device):
    model.eval()

    subject_names = [subject['name'] for subject in subjects]
    print(f"generating segmentations for subjects: {subject_names}")

    with torch.no_grad():
        subjects, _ = predictor.predict(model=model, device=device, subjects=subjects)

    for subject in subjects:
        transform = subject.get_composed_history()
        inverse_transform = transform.inverse(warn=False)
        pred_subject = tio.Subject({"y": subject["y_pred"]})
        inverse_pred_subject = inverse_transform(pred_subject)
        output_label = inverse_pred_subject.get_first_image()
        subject["y_pred"].set_data(output_label["data"].to(torch.int32))

    return subjects


def post_process(output_label):

    label_data = output_label["data"][0].numpy()

    label_data, hole_voxels_removed = remove_holes(label_data, hole_size=64)
    txt_output = f"Filled {hole_voxels_removed} voxels from detected holes.\n"

    num_components = label_data.max()
    label_data, num_components_removed, num_elements_removed = keep_components(label_data, num_components)
    txt_output += f"Removed {num_elements_removed} voxels from {num_components_removed} components."

    label_data = torch.from_numpy(label_data[None]).to(torch.int32)
    output_label.set_data(label_data)

    return txt_output


def save_subjects_img(subjects, image_name, out_folder, output_filename):
    for subject in subjects:

        if out_folder == "":
            out_folder_path = Path(subject["folder"])
        else:
            out_folder_path = Path(out_folder) / "subjects" / subject["name"]

        out_folder_path.mkdir(exist_ok=True, parents=True)

        subject[image_name].save(out_folder_path / (output_filename + ".nii.gz"))


def post_process_subjects(subjects, image_name):
    txt_output = ""
    for subject in subjects:
        txt_output += subject["name"] + "\n"
        txt_output += post_process(subject[image_name]) + "\n"

    return txt_output


def main(
    dataset_path: str,
    out_folder: str,
    output_filename: str,
    num_workers: int = 0,
    batch_size: int = 4,
    cpu: bool = False,
):
    """Auto Hippocampus Segmentation  Run with `python predict.py` followed by args

    Args:
        dataset_path: Path to the subjects data folders. See readme.md for directory structure.
        out_folder: Folder for output.
        output_filename: File name for segmentation output. Provided extensions will be ignored and file will be saved ass .nii.gz.
        num_workers: How many CPU threads to use for data loading, preprocessing, and augmentation.
        batch_size: How many subjects should be run through the model at once. If memory issues, reduce batch size.
        cpu: Sets the pytorch device to use cpu. Use if there are issues with gpu.
    """


    if not cpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else: 
            device = torch.device("cpu")
            print("cuda not available, switched to cpu")
    else:
        device = torch.device("cpu")


    print("using device", device)

    roi_union_path = Path(__file__).parent / "segmentation_pipeline/atlas/whole_roi_union.nii.gz"

    ensemble_path = Path(__file__).parent / "saved_models/hippo/standard"
    contexts = []
    for file_path in ensemble_path.iterdir():
        context = TorchContext(device, file_path=file_path, variables=dict(DATASET_PATH=dataset_path))
        context.keep_components(("model", "trainer", "dataset"))

        context.component_definitions[0]['params']['subject_loader'].loaders[7] = ImageLoader(
            glob_pattern=str(roi_union_path),
            image_name="whole_roi_union",
            image_constructor=tio.LabelMap, uniform=True
        )

        context.init_components()

        context.model = EnsembleFlips(context.model, strategy="majority", spatial_dims=(3, 4))

        contexts.append(context)
    print("Loaded models.")


    models = [context.model for context in contexts]
    context.model = EnsembleModels(models, strategy="majority")

    dataset = context.dataset
    dataset.transform.transforms[0].transforms.insert(0, ReplaceNan())

    dataloader = context.trainer.validation_dataloader_factory.get_data_loader(
        dataset=dataset, batch_size=batch_size, num_workers=num_workers
    )

    for subjects in dataloader:

        subjects = inference(subjects, context.trainer.validation_predictor, context.model, device)
        _ = post_process_subjects(subjects, "y_pred")

        save_subjects_img(subjects, 'y_pred', out_folder, output_filename)

if __name__ == "__main__":
    fire.Fire(main)
