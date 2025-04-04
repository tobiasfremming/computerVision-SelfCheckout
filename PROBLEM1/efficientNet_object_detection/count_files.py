import os

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

if __name__ == "__main__":
    import sys

    # Get directory from command line or use current directory
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd()

    if not os.path.exists(path):
        print(f"Directory '{path}' does not exist.")
    else:
        total_files = count_files(path)
        print(f"Total number of files in '{path}' and its subdirectories: {total_files}")