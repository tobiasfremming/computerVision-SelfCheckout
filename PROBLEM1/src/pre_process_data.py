import os
import shutil
import json
from PIL import Image

def split_and_crop_bb_images(
    source_dir, 
    dest_dir, 
    validation_dir, 
    image_interval=5
):
    """
    Finds images with '_bb' in the filename, reads the corresponding
    bounding box .txt file (with '_bb' removed), crops according to UV
    coordinates, and saves images to training or validation sets.

    Args:
        source_dir (str): Source directory of images.
        dest_dir (str): Destination directory for training images.
        validation_dir (str): Destination directory for validation images.
        image_interval (int): Every nth cropped image is added to validation,
                              others go to training.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created training destination directory: {dest_dir}")
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
        print(f"Created validation destination directory: {validation_dir}")

    total_files = 0
    train_copied = 0
    valid_copied = 0
    valid_image_index = 0
    dirs_processed = 0

    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        train_target_dir = os.path.join(dest_dir, relative_path) if relative_path != '.' else dest_dir
        valid_target_dir = os.path.join(validation_dir, relative_path) if relative_path != '.' else validation_dir

        os.makedirs(train_target_dir, exist_ok=True)
        os.makedirs(valid_target_dir, exist_ok=True)
        dirs_processed += 1

        # Sort files for consistency
        files = sorted(files)

        for filename in files:
            # We only care about images with '_bb' in their name (and not .txt)
            if '_bb' not in filename or filename.endswith('.txt'):
                continue

            # Found a bounding-box image
            total_files += 1
            src_path = os.path.join(root, filename)

            # Derive the corresponding text filename by removing '_bb'
            txt_name = filename.replace('_bb', '').rsplit('.', 1)[0] + '.txt'
            txt_path = os.path.join(root, txt_name)

            if not os.path.isfile(txt_path):
                # If the matching bounding box file is missing, skip
                print(f"No matching bounding box file for {src_path}. Expected {txt_path}. Skipping.")
                continue

            # Load bounding box data
            with open(txt_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {txt_path}: {e}")
                    continue

            # Open the '_bb' image
            try:
                image = Image.open(src_path)
            except Exception as e:
                print(f"Error opening {src_path}: {e}")
                continue

            width, height = image.size  # We'll rely on the actual image size

            # 'label' field is an array of bounding boxes
            if "label" not in data:
                print(f"No 'label' field in {txt_path}, skipping.")
                continue

            bounding_boxes = data["label"]
            if not isinstance(bounding_boxes, list):
                print(f"'label' in {txt_path} is not a list. Skipping.")
                continue

            # Process each bounding box
            for idx, box in enumerate(bounding_boxes):
                # Make sure all needed keys exist
                if not all(k in box for k in ("topX", "topY", "bottomX", "bottomY")):
                    print(f"Missing bounding box keys in {txt_path} for index {idx}, skipping.")
                    continue

                topX = box["topX"]
                topY = box["topY"]
                bottomX = box["bottomX"]
                bottomY = box["bottomY"]

                # Convert UV coords to pixel coords
                left = int(round(topX * width))
                upper = int(round(topY * height))
                right = int(round(bottomX * width))
                lower = int(round(bottomY * height))

                # Ensure valid coords
                left = max(0, left)
                upper = max(0, upper)
                right = min(width, right)
                lower = min(height, lower)

                if left >= right or upper >= lower:
                    print(f"Invalid bounding box in {txt_path} for index {idx}, skipping crop.")
                    continue

                # Crop the image
                cropped_img = image.crop((left, upper, right, lower))

                # Determine output file name
                # e.g. "example_bb.png" -> "example_bb_crop0.png"
                name_without_ext, ext = os.path.splitext(filename)
                out_filename = f"{name_without_ext}_crop{idx}{ext}"

                # Decide whether this crop goes to validation or train
                if valid_image_index % image_interval == 0:
                    # Save to validation
                    dst_valid_path = os.path.join(valid_target_dir, out_filename)
                    cropped_img.save(dst_valid_path)
                    valid_copied += 1
                else:
                    # Save to training
                    dst_train_path = os.path.join(train_target_dir, out_filename)
                    cropped_img.save(dst_train_path)
                    train_copied += 1

                valid_image_index += 1

                # Progress info
                if (train_copied + valid_copied) % 100 == 0 and (train_copied + valid_copied) > 0:
                    print(f"Progress: {train_copied} train, {valid_copied} validation cropped images saved...")

    print(f"\nCompleted!")
    print(f"Directories processed: {dirs_processed}")
    print(f"Total bb-image files evaluated: {total_files}")
    print(f"Training crops saved: {train_copied}")
    print(f"Validation crops saved: {valid_copied}")


if __name__ == "__main__":
    # Example usage:
    source_directory = "../../data/images/NGD_HACK"
    train_directory = "images/NGD_HACK_TRAIN_CROPPED"
    validation_directory = "images/NGD_HACK_VALIDATION_CROPPED"

    print(f"Starting processing from {source_directory}")
    print(f"Training images -> {train_directory}")
    print(f"Validation images -> {validation_directory} (every 5th cropped image)\n")

    split_and_crop_bb_images(
        source_dir=source_directory,
        dest_dir=train_directory,
        validation_dir=validation_directory,
        image_interval=5
    )
