.PHONY: *

PYTHON_EXEC := python3.10


CLEARML_PROJECT_NAME := lidar_pcl_features
CLEARML_DATASET_NAME := lidar_cvpr09

DATASET_TEMP_DIR := .data_temp_dir
UNPACKED_DATASET_DIR := $(DATASET_TEMP_DIR)/$(CLEARML_DATASET_NAME)


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


jupyterlab_start:
	# These lines ensure that CTRL+B can be used to jump to definitions in
	# code of installed modules.
	# Explained here: https://github.com/jupyter-lsp/jupyterlab-lsp/blob/39ee7d93f98d22e866bf65a80f1050d67d7cb504/README.md?plain=1#L175
	ln -s / .lsp_symlink || true  # Create if does not exist.
	jupyter nbextension enable --py --sys-prefix widgetsnbextension
	jupyter lab --ContentsManager.allow_hidden=True


migrate_dataset:
	# Migrate dataset to ClearML datasets.
	rm -R $(DATASET_TEMP_DIR) || true
	@make get_data
	@make unpack_data
	@make upload_dataset_to_clearml
	rm -R $(DATASET_TEMP_DIR)


get_data:
	mkdir -p $(UNPACKED_DATASET_DIR)

	# Test and train switched intentionally
	wget https://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/data/testing.zip -O $(DATASET_TEMP_DIR)/testing.zip
	wget https://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/data/training.zip -O $(DATASET_TEMP_DIR)/training.zip
	wget https://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/filtering_mapping.txt -O $(UNPACKED_DATASET_DIR)/classes.txt


unpack_data:
	unzip $(DATASET_TEMP_DIR)/training.zip -d $(UNPACKED_DATASET_DIR) || true	# Ignore errors
	unzip $(DATASET_TEMP_DIR)/testing.zip -d $(UNPACKED_DATASET_DIR) || true	# Ignore errors


upload_dataset_to_clearml:
	clearml-data create --project $(CLEARML_PROJECT_NAME) --name $(CLEARML_DATASET_NAME)
	find $(DATASET_TEMP_DIR) -type f -name '.DS_Store' -delete
	clearml-data add --files $(UNPACKED_DATASET_DIR)
	clearml-data close
