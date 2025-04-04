import os
import shutil

def split_images_into_train_and_validation(source_dir, dest_dir, validation_dir, image_interval=5):
    """
    Copy all non-bounding-box, non-.txt images from source_dir.
    Every `image_interval`th image is copied to validation_dir.
    All other images are copied to dest_dir.
    Original folder structure is preserved.

    Args:
        source_dir (str): Source directory of images.
        dest_dir (str): Destination directory for training images.
        validation_dir (str): Destination directory for validation images.
        image_interval (int): Every nth image is added to validation only.
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
            total_files += 1

            if '_bb' in filename or filename.endswith('.txt'):
                continue

            src_path = os.path.join(root, filename)

            try:
                if valid_image_index % image_interval == 0:
                    # Copy to validation only
                    dst_valid_path = os.path.join(valid_target_dir, filename)
                    shutil.copy2(src_path, dst_valid_path)
                    valid_copied += 1
                else:
                    # Copy to training only
                    dst_train_path = os.path.join(train_target_dir, filename)
                    shutil.copy2(src_path, dst_train_path)
                    train_copied += 1
            except Exception as e:
                print(e)

            valid_image_index += 1

            if (train_copied + valid_copied) % 100 == 0:
                print(f"Progress: {train_copied} train, {valid_copied} validation images copied...")

    print(f"\nCompleted!")
    print(f"Directories processed: {dirs_processed}")
    print(f"Total image files evaluated: {valid_image_index}")
    print(f"Training images copied: {train_copied}")
    print(f"Validation images copied: {valid_copied}")


if __name__ == "__main__":
    # Set these to your actual paths
    source_directory = "images/NGD_HACK"
    train_directory = "images/NGD_HACK_NO_BB"
    validation_directory = "images/NGD_HACK_VALIDATION"

    print(f"Starting processing from {source_directory}")
    print(f"Training images -> {train_directory}")
    print(f"Validation images -> {validation_directory} (every 5th image)\n")

    split_images_into_train_and_validation(source_directory, train_directory, validation_directory, image_interval=5)