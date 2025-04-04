import os
import shutil
from pathlib import Path

def copy_non_bb_files_recursive(source_dir, dest_dir):
    """
    Recursively copy all files from source_dir to dest_dir that don't have '_bb' in their filename,
    preserving the original folder structure.
    
    Args:
        source_dir (str): Source directory containing the files
        dest_dir (str): Destination directory where files will be copied
    """
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")
    
    # Counters for statistics
    total_files = 0
    copied_count = 0
    dirs_processed = 0
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Create the same directory structure in the destination
        relative_path = os.path.relpath(root, source_dir)
        if relative_path == '.':
            target_dir = dest_dir
        else:
            target_dir = os.path.join(dest_dir, relative_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
        
        dirs_processed += 1
        
        # Process files in the current directory
        for filename in files:
            total_files += 1
            
            # Check if the file doesn't have '_bb' in its name
            if '_bb' not in filename and '.txt' not in filename:
                src_path = os.path.join(root, filename)
                dst_path = os.path.join(target_dir, filename)
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                
                # Print progress every 100 files
                if copied_count % 100 == 0:
                    print(f"Progress: Copied {copied_count} files out of {total_files} processed...")
    
    print(f"\nCompleted!")
    print(f"Directories processed: {dirs_processed}")
    print(f"Total files examined: {total_files}")
    print(f"Files copied (without '_bb'): {copied_count}")

if __name__ == "__main__":
    # Replace these with your actual source and destination directories
    source_directory = "images/NGD_HACK"
    destination_directory = "images/NGD_HACK_NO_BB"
    
    print(f"Starting to copy files without '_bb' from {source_directory} to {destination_directory}")
    print(f"This will preserve the original folder structure...")
    copy_non_bb_files_recursive(source_directory, destination_directory)