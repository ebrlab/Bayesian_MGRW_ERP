import glob
import mne
import pickle

info_path = "info.pickle"
with open(info_path, 'rb') as handle:
    info = pickle.load(handle)

if isinstance(info, mne.Info):
    # Save the Info object to a .fif file
    output_fif_path = "info.fif"
    mne.io.write_info(output_fif_path, info)
    print(f"Info has been saved to {output_fif_path}")
else:
    print("The loaded object is not an MNE Info object.")