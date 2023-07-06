import os

directory = "sample_data"

def rename_file(old_filename):
    # Replace spaces with no spaces
    new_filename = old_filename.replace(" ", "")
    old_path = os.path.join(directory, old_filename)
    new_path = os.path.join(directory, new_filename)
    print(old_filename)
    print(new_filename)

    # Rename the file
    os.rename(old_path, new_path)

    return new_filename




def rename_all_files():
    for file in os.listdir(directory):
        rename_file(file)
        
if __name__ == "__main__":
    rename_all_files()

