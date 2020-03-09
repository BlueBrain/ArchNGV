
ARCHNGV_PATH=$(dirname $(dirname $(readlink -fm "$0")))
ARCHNGV_EXEMPLAR="$ARCHNGV_PATH/exemplar/template"

TARGET_PARENT_DIR=$1

for INDEX in 0 1 2 3 4 5 6
do

    TARGET_DIR="$TARGET_PARENT_DIR/exemplar_$INDEX"

    mkdir -p $TARGET_DIR

    ARCHNGV_SNAKEFILE="$ARCHNGV_PATH/snakemake/Snakefile"
    TARGET_SNAKEFILE="$TARGET_DIR/Snakefile"

    ARCHNGV_BIONAME="$ARCHNGV_EXEMPLAR/bioname"
    TARGET_BIONAME="$TARGET_DIR/bioname"

    ln -s $ARCHNGV_BIONAME $TARGET_BIONAME
    ln -s $ARCHNGV_SNAKEFILE $TARGET_SNAKEFILE
    ln -s "$ARCHNGV_EXEMPLAR/run.sh" "$TARGET_DIR/run.sh" && chmod +x "$TARGET_DIR/run.sh"

done
