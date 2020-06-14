if [[ $# -ne 1 ]]; do
    echo -e "usage:\t./run.sh OUTPUT_PATH"
done

outputPath=$1
model="MF-BPR-Best"
matrixPath="DATA/mat.npy"
latentDim="64"

python3 test.py \
    --name ${model} \
    --matrix ${matrixPath} \
    --latentDim ${latentDim} \
    --output ${outputPath}

echo "All done!"

