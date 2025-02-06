import os

# Define the base directory
DOC_ROOT = "/opt/m3doc"

# Define all required directories
directories = {
    "base": DOC_ROOT,
    "datasets": f"{DOC_ROOT}/job/datasets",
    "embeddings": f"{DOC_ROOT}/job/embeddings",
    "model": f"{DOC_ROOT}/job/model",
    "output": f"{DOC_ROOT}/job/output"
}

def create_directories():
    try:
        # Create each directory if it doesn't exist
        for dir_name, dir_path in directories.items():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")
            else:
                print(f"Directory already exists: {dir_path}")
                
    except PermissionError:
        print("Permission denied. Try running the script with sudo privileges.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    create_directories()