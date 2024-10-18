import h5py
import h5py

# def open_hdf5_file(file_path):
#   try:
#     with h5py.File(file_path, 'r') as f:
#       # Do something with the HDF5 file
#       print(f.keys())  # List dataset names
#   except FileNotFoundError as e:
#     print(f"Error opening HDF5 file: {e}")

# # Replace 'path/to/your/file.h5' with the correct path
# file_path = '/scratch/users/u11644/data/pannuke/pannuke_fold_2.h5'
# open_hdf5_file(file_path)



# def explore_hdf5_structure(filename):
#   with h5py.File(filename, 'r') as f:
#     def print_structure(name, obj):
#       print(name)
#       if isinstance(obj, h5py.Group):
#         for key, value in obj.items():
#           print_structure(f"{name}/{key}", value)
#       elif isinstance(obj, h5py.Dataset):
#         print(f"  shape: {obj.shape}")
#         print(f"  dtype: {obj.dtype}")
#         print(f"  compression: {obj.compression}")
#     print_structure("/", f)

# # Example usage:
# explore_hdf5_structure("pannuke_fold_2.h5")


from PIL import Image

# def visualize_tiff(tiff_file_path):
#   """Visualizes a TIFF file using Pillow.

#   Args:
#     tiff_file_path: Path to the TIFF file.
#   """

#   try:
#     img = Image.open(tiff_file_path)
#     img.show()
#   except Exception as e:
#     print(f"Error opening TIFF file: {e}")

# # Example usage:
    

def print_tiff_shape(tiff_file_path):
  try:
    img = Image.open(tiff_file_path)
    print(img.size)  # This will print the width and height as a tuple
    img.show()
  except Exception as e:
    print(f"Error opening TIFF file: {e}")


# Example usage:
# image_path = "path/to/your/image.tiff"
image_path = "/scratch/users/u11644/data/pannuke/fold2_as_tiff/images/image_430.tiff"

print_tiff_shape(image_path)
# visualize_tiff(image_path)