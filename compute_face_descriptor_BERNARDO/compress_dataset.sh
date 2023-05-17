#!/bin/bash

CURR_PATH=$(pwd)

# specify the directory to compress
# DIR=/path/to/directory
# DIR=/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_22subj
DIR=/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw
# DIR=/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images

BASENAME_DIR=$(basename $DIR)
PARENT_DIR=$(dirname $DIR)
cd $PARENT_DIR

# create a temporary file to store the list of files to compress
# FILELIST=$(mktemp)
FILELIST=./files_to_compress.txt

# find all "mesh_centralized-nosetip_with-normals_filter-radius=100.npy" files within the directory and its subdirectories, and write their paths to the temporary file
find "$BASENAME_DIR" -type f -name "mesh_centralized-nosetip_with-normals_filter-radius=100.npy" > "$FILELIST"

# create a tar archive of all files listed in the temporary file, and compress it using gzip
echo "Compressing files from \""$FILELIST"\" ..."
tar -czvf $BASENAME_DIR.tar.gz -T "$FILELIST"

# delete the temporary file
rm "$FILELIST"
cd $CURR_PATH
