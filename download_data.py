from ultralytics.data.utils import check_det_dataset

info = check_det_dataset("VisDrone.yaml")
print("Dataset ready at:", info["path"])
print("Classes:", info["names"])