import kagglehub
import shutil
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300")
print("Dataset downloaded to:", path)

current_dir = os.getcwd()

for filename in os.listdir(path):
    src_file = os.path.join(path, filename)
    dst_file = os.path.join(current_dir, filename)
    
    if os.path.isfile(src_file):
        print(f"Moving {filename} to {current_dir}")
        shutil.move(src_file, dst_file)

print("Files moved to current directory:", current_dir)
print("Files in current directory:", os.listdir(current_dir))